import os
import argparse
import numpy as np
import torch
from torch.utils import data
import torch.multiprocessing.spawn
from torch.utils.tensorboard import SummaryWriter
from config.config import _C as cfg
import util.distributed as du
from dataset import get_dataset
from generation import generation
from utils import initialize_training, model_epoch_new, auto_convergence, analyse_inputs, setup_logger


def main(local_rank=-1):
    parser = argparse.ArgumentParser(description="MAP Training")
    parser.add_argument(
        "--config-file",
        default="config/train_water.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)

    # Set the local rank for distributed training
    if local_rank != -1: os.environ['LOCAL_RANK'] = str(local_rank)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    local_world_size = min(torch.cuda.device_count(), cfg.NUM_GPUS)

    model, optimizer = initialize_training(cfg)
    train_dataset, test_dataset, dataDims = get_dataset(cfg)
    # distributed data parallelism
    if local_world_size > 1:
        train_sampler = data.DistributedSampler(train_dataset)
        tr = data.DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=train_sampler, pin_memory=True)
    else:
        tr = data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    te = data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    if du.is_master_proc(num_gpus=local_world_size):
        writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR)
        logger = setup_logger(os.path.join(cfg.OUTPUT_DIR, 'log.txt'))
        logger.info("Train with config:")
        logger.info(cfg)
        num_trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        logger.info(f"Number of trainable params: {num_trainable_params}")
        # input_analysis = analyse_inputs(cfg, dataDims, dataset=tr.dataset)
        # logger.info(f'Input Analysis:\n{input_analysis}')
        # logger.info('Imported and Analyzed Training Dataset {}'.format(cfg.training_dataset))
    else:
        writer = None
        logger = None

    epoch = 1
    converged = 0
    tr_err_hist = []
    te_err_hist = []

    scaler = torch.amp.GradScaler("cuda" if cfg.CUDA else "cpu") if cfg.ENABLE_AMP else None

    while epoch <= cfg.max_epochs and not converged:
        if local_world_size > 1:
            train_sampler.set_epoch(epoch - 1) # shuffle the training data at each epoch
        err_tr, time_tr = model_epoch_new(cfg, 
                                          dataDims=dataDims, 
                                          trainData=tr, 
                                          model=model, 
                                          optimizer=optimizer, 
                                          update_gradients=True,
                                          epoch=epoch,
                                          scaler=scaler,
                                          tb_writer=writer,
                                          iteration_override=0)  # train & compute loss
        err_te, time_te = model_epoch_new(cfg, 
                                          dataDims=dataDims, 
                                          trainData=te, 
                                          model=model, 
                                          update_gradients=False,
                                          epoch=epoch,
                                          scaler=scaler,
                                          tb_writer=writer,
                                          iteration_override=0)  # compute loss on test set
        tr_err_hist.append(torch.mean(torch.stack(err_tr)))
        te_err_hist.append(torch.mean(torch.stack(err_te)))
        converged = auto_convergence(cfg, epoch, tr_err_hist, te_err_hist, logger)

        if du.is_master_proc(num_gpus=local_world_size):
            out_str = 'epoch={}; nll_tr={:.5f}; nll_te={:.5f}; time_tr={:.1f}s; time_te={:.1f}s'.format(epoch, tr_err_hist[-1], te_err_hist[-1], time_tr, time_te)
            logger.info(out_str)
            # Logging with TensorBoard
            writer.add_scalar("Loss/train", tr_err_hist[-1], epoch-1)
            writer.add_scalar("Loss/test", te_err_hist[-1], epoch-1)
            # Save the model state
            if epoch % cfg.ckpt_save_period == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if local_world_size > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    "cfg": cfg}, os.path.join(cfg.OUTPUT_DIR, f'model-ep{epoch}.pt'))
        epoch += 1
    if du.is_master_proc(num_gpus=local_world_size):
        torch.save({
            'epoch': epoch-1,
            'model_state_dict': model.module.state_dict() if local_world_size > 1 else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            "cfg": cfg}, os.path.join(cfg.OUTPUT_DIR, f'model-last-ep{epoch-1}.pt'))
        writer.close()
        logger.info('Training finished!')
        sample, time_ge = generation(cfg, dataDims, model)
        # log_generation_stats(cfg, epoch, experiment, sample, agreements, output_analysis)



if __name__ == "__main__":
    ngpus = min(torch.cuda.device_count(), cfg.NUM_GPUS)
    if ngpus > 1:
        print(f"Starting distributed training with {ngpus} GPUs...")
        torch.multiprocessing.spawn(main, args=(), nprocs=ngpus, join=True)
    else:
        main()
