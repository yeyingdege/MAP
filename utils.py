# metrics to determine the performance of our learning algorithm
from comet_ml import Experiment
import os
import time
import pickle
import sys
import numpy as np
from tqdm import tqdm
from scipy.fft import fftn, fftshift
from scipy.ndimage import gaussian_filter
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch import optim, cuda
from torch.utils import data
from torch.nn.parallel import DistributedDataParallel as DDP
from models import GatedPixelCNN
from dataset import build_dataset
import util.distributed as du
# from accuracy_metrics import *
# from Image_Processing_Utils import *



def get_dir_name(model, training_data, filters, layers, dilation, filter_size, noise, den_var, dataset_size):
    dir_name = "model=%d_dataset=%d_dataset_size=%d_filters=%d_layers=%d_dilation=%d_filter_size=%d_noise=%.1f_denvar=%.1f" % (model, training_data, dataset_size, filters, layers, dilation, filter_size, noise, den_var)  # directory where tensorboard logfiles will be saved

    return dir_name


def get_model(cfg):
    if cfg.model == 'gated1':
        return GatedPixelCNN(cfg) # gated, without blind spot
    sys.exit("Unknown model type.")


def initialize_training(cfg):
    model = get_model(cfg)
    if cfg.CUDA:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        local_world_size = min(torch.cuda.device_count(), cfg.NUM_GPUS)
        # Determine the GPU used by the current process
        cur_device = torch.cuda.current_device()
        du.init_process_group(
            cur_device,
            local_world_size,
            shard_id=0,
            num_shards=1,
            init_method=cfg.init_method
        )
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[cur_device],
            output_device=cur_device,
            find_unused_parameters=True
        )
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, nesterov=True)
    # optimizer = optim.AdamW(model.parameters(), lr=configs.lr, amsgrad=True)
    return model, optimizer


def compute_loss(output, target):
    target = target[:,:1]
    return F.cross_entropy(output, target.squeeze(1).long())


def get_training_batch_size(configs, model):
    finished = 0
    training_batch_0 = 1 * configs.training_batch_size
    #  test various batch sizes to see what we can store in memory
    dataset = build_dataset(configs)
    dataDims = dataset.dataDims
    train_size = int(0.8 * len(dataset))  # split data into training and test sets
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.Subset(dataset, [range(train_size),range(train_size,test_size + train_size)])  # split it the same way every time
    optimizer = optim.SGD(model.parameters(),lr=1e-2, momentum=0.9, nesterov=True)#optim.AdamW(model.parameters(),amsgrad=True) #

    while (configs.training_batch_size > 1) & (finished == 0):
        try:
            test_dataloader = data.DataLoader(test_dataset, batch_size=configs.training_batch_size, shuffle=False, num_workers=0, pin_memory=True)
            model_epoch(configs, dataDims = dataDims, trainData = test_dataloader, model = model, optimizer = optimizer, update_gradients = True, iteration_override = 2)
            finished = 1
        except RuntimeError: # if we get an OOM, try again with smaller batch
            configs.training_batch_size = int(np.ceil(configs.training_batch_size * 0.8)) - 1
            print('Training batch sized reduced to {}'.format(configs.training_batch_size))

    return max(int(configs.training_batch_size * 0.25),1), int(configs.training_batch_size != training_batch_0)


def model_epoch(configs, dataDims = None, trainData = None, model=None, optimizer=None, update_gradients = True, iteration_override = 0):
    if configs.CUDA:
        cuda.synchronize()  # synchronize for timing purposes
    time_tr = time.time()

    err = []

    if update_gradients:
        model.train(True)
    else:
        model.eval()
    print(['traindata',len(trainData)])
    for i, input in enumerate(trainData):

        if configs.CUDA:
            input = input.cuda(non_blocking=True)

        target = input * dataDims['classes']

        output = model(input.float()) # reshape output from flat filters to channels * filters per channel
        loss = compute_loss(output, target)

        err.append(loss.data)  # record loss

        if update_gradients:
            optimizer.zero_grad()  # reset gradients from previous passes
            loss.backward()  # back-propagation
            optimizer.step()  # update parameters

        if iteration_override != 0:
            if i > iteration_override:
                break

    print(i)
    if configs.CUDA:
        cuda.synchronize()
    time_tr = time.time() - time_tr

    return err, time_tr

def model_epoch_new(configs, dataDims, trainData, model, optimizer=None, epoch=0, tb_writer=None, update_gradients=True, iteration_override=0):
    if configs.CUDA:
        cuda.synchronize()  # synchronize for timing purposes
    time_tr = time.time()
    err = []

    if update_gradients:
        model.train()
        mode = 'train'
    else:
        model.eval()
        mode = 'test'
    print(mode, len(trainData))

    for i, input in enumerate(tqdm(trainData)):
        if configs.CUDA:
            input = input.cuda(non_blocking=True)
        # target = (input * dataDims['classes']) # [1, 1, 40, 40, 40], label: 1, 2, 3
        target = (input.clone() * dataDims['classes'] - 1) # [1, 1, 40, 40, 40], label: 0, 1, 2

        output = model(input.float()) # reshape output from flat filters to channels * filters per channel
        loss = compute_loss(output, target) # output: [1, 3, 40, 40, 40]
        err.append(loss.data)  # record loss

        if update_gradients:
            optimizer.zero_grad()  # reset gradients from previous passes
            loss.backward()  # back-propagation
            optimizer.step()  # update parameters

        global_step = (epoch - 1) * len(trainData) + i
        if tb_writer is not None:
            tb_writer.add_scalar(f"Loss/{mode}_batch", err[-1], global_step)

        if iteration_override != 0 and i >= iteration_override: break

    if configs.CUDA:
        cuda.synchronize()
    time_tr = time.time() - time_tr
    return err, time_tr


def auto_convergence(configs, epoch, tr_err_hist, te_err_hist):
    # set convergence criteria
    # if the test error has increased on average for the last x epochs
    # or if the training error has decreased by less than 1% for the last x epochs
    #train_margin = .000001  # relative change over past x runs
    # or if the training error is diverging from the test error by more than 20%
    test_margin = 10 # max divergence between training and test losses
    # configs.convergence_moving_average_window - the time over which we will average loss in order to determine convergence
    converged = 0
    if epoch > configs.convergence_moving_average_window:
        print('hi')
        window = configs.convergence_moving_average_window
        tr_mean, te_mean = [torch.mean(torch.stack(tr_err_hist[-configs.convergence_moving_average_window:])), torch.mean(torch.stack(te_err_hist[-configs.convergence_moving_average_window:]))]
        print([tr_mean,tr_err_hist[-window],configs.convergence_margin,te_mean])
        if (torch.abs((tr_mean - tr_err_hist[-window]) / tr_mean) < configs.convergence_margin) \
                or ((torch.abs(te_mean - tr_mean) / tr_mean) < configs.convergence_margin) \
                or (epoch == configs.max_epochs)\
                or (te_mean > te_err_hist[-window]):
            converged = 1
            print('Learning converged at epoch {}'.format(epoch - window))  # print a nice message  # consider also using an accuracy metric

    return converged


def analyse_inputs(configs, dataDims, dataset=None):
    if dataset is None:
        dataset = torch.Tensor(build_dataset(configs))  # get data
    dataset = dataset * (dataDims['classes'])-1
    dataset = dataset[0:10]
    input_analysis = analyse_samples(dataset)
    input_analysis['training samples'] = dataset[0:10,0]
    return input_analysis


def density(voxel):
    return torch.tensor(np.sum(voxel) / voxel.size)

def paircorrelation3d_lattice(voxel, bins, rmax, dr, rho):
    coords = np.argwhere(voxel > 0)
    n_points = len(coords)
    if n_points < 2:
        return torch.tensor(0.), torch.tensor(np.zeros(bins))

    distances = []
    for i in range(n_points):
        for j in range(i + 1, n_points):
            d = np.linalg.norm(coords[i] - coords[j])
            if d < rmax:
                distances.append(d)

    hist, bin_edges = np.histogram(distances, bins=bins, range=(0, rmax))
    shell_volumes = 4/3 * np.pi * ((bin_edges[1:]**3) - (bin_edges[:-1]**3))
    expected = rho * shell_volumes * n_points
    g_r = hist / expected
    g_r = np.nan_to_num(g_r)

    return torch.tensor(np.mean(g_r)), torch.tensor(g_r)


def fourier3d_radial(voxel):
    fft_voxel = np.abs(fftshift(fftn(voxel)))**2
    center = np.array(fft_voxel.shape) // 2
    z, y, x = np.indices(fft_voxel.shape)
    r = np.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)

    r = r.astype(np.int32)
    max_r = r.max()
    radial_sum = np.bincount(r.ravel(), weights=fft_voxel.ravel(), minlength=max_r+1)
    radial_count = np.bincount(r.ravel(), minlength=max_r+1)
    radial_profile = radial_sum / np.maximum(radial_count, 1)

    return torch.tensor(radial_profile)


def analyse_samples(sample):
    sample = sample.squeeze(1)
    xrange_list = []
    yrange_list = []
    zrange_list = []

    for i in range(len(sample)):
        indices = np.argwhere(sample[i] == 1)
        if len(indices) > 0:
            x, y, z = indices[:, 0], indices[:, 1], indices[:, 2]
            xrange_list.extend(x)
            yrange_list.extend(y)
            zrange_list.extend(z)

    if xrange_list:
        xrange = max(xrange_list) - min(xrange_list)
        yrange = max(yrange_list) - min(yrange_list)
        zrange = max(zrange_list) - min(zrange_list)
    else:
        xrange = yrange = zrange = 0
    # print([xrange, yrange, zrange])
    maxrange = max(xrange, yrange, zrange)
    correlationrange = maxrange / 3 if maxrange > 0 else 1
    dr = 0.5
    rho = 0.0078

    exdens = []
    list_corr = []
    list_rcorr = []
    list_fourier = []

    for i in range(len(sample)):
        exdens.append(density(sample[i]))
        corr, rcorr = paircorrelation3d_lattice(sample[i], 10, correlationrange, dr, rho)
        list_corr.append(corr)
        list_rcorr.append(rcorr)
        list_fourier.append(fourier3d_radial(sample[i]))

    sample_analysis = {
        'density': torch.mean(torch.stack(exdens)).item(),
        'radial correlation': sum(list_corr).item() / len(list_corr),
        'correlation bins': torch.mean(torch.stack(list_rcorr), dim=0).numpy(),
        'radial fourier': torch.mean(torch.stack(list_fourier), dim=0).numpy()
    }

    return sample_analysis


def compute_accuracy(configs, dataDims, input_analysis, output_analysis):

    #input_xdim, input_ydim, sample_xdim, sample_ydim = [input_analysis['fourier2d'].shape[-1], input_analysis['fourier2d'].shape[-2], output_analysis['fourier2d'].shape[-1], output_analysis['fourier2d'].shape[-2]]

    #input_xdim, input_ydim, sample_xdim, sample_ydim = [input_analysis['correlation2d'].shape[-1], input_analysis['correlation2d'].shape[-2], output_analysis['correlation2d'].shape[-1], output_analysis['correlation2d'].shape[-2]]
    #if configs.sample_outpaint_ratio > 1: # shrink inputs to meet outputs or vice-versa
    #    x_difference = sample_xdim-input_xdim
    #    y_difference = sample_ydim-input_ydim
    #   output_analysis['correlation2d'] = output_analysis['correlation2d'][y_difference//2:-y_difference//2, x_difference//2:-x_difference//2]
    #elif configs.sample_outpaint_ratio < 1:
    #    x_difference = input_xdim - sample_xdim
    #    y_difference = input_ydim- sample_ydim
    #    input_analysis['correlation2d'] = input_analysis['correlation2d'][y_difference // 2:-y_difference // 2, x_difference // 2:-x_difference // 2]

    agreements = {}
    agreements['density'] = np.amax((1 - np.abs(input_analysis['density'] - output_analysis['density']) / input_analysis['density'],0))
    #agreements['correlation'] = np.amax((1 - np.sum(np.abs(input_analysis['correlation2d'] - output_analysis['correlation2d'])) / (np.sum(input_analysis['correlation2d']) + 1e-8),0))

    return agreements


def save_ckpt(epoch, net, optimizer, dir_name):
    torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, 'ckpts/' + dir_name[:])

def load_all_pickles(path):
    outputs = []
    print('loading all .pkl files from',path)
    files = [ f for f in listdir(path) if isfile(join(path,f)) ]
    for f in files:
        if f[-4:] in ('.pkl'):
            name = f[:-4]+'_'+f[-3:]
            print('loading', f, 'as', name)
            with open(path + '/' + f, 'rb') as f:
                outputs.append(pickle.load(f))

    return outputs

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def find_dir():
    found = 0
    ii = 0
    while found == 0:
        ii += 1
        if not os.path.exists('logfiles/run_%d' % ii):
            found = 1

    return ii

def rolling_mean(input, run):
    output = np.zeros(len(input))
    for i in range(len(output)):
        if i < run:
            output[i] = np.average(input[0:i])
        else:
            output[i] = np.average(input[i - run:i])

    return output



def get_comet_experiment(configs):
    if configs.comet:
        # Create an experiment with your api key
        experiment = Experiment(
            api_key="WdZXLSYozVLDkUZWGfcLPj1pu",
            project_name="wled",
            workspace="ata-madanchi",
        )
        experiment.set_name(configs.experiment_name + str(configs.run_num))
        experiment.log_metrics(configs.__dict__)
        experiment.log_others(configs.__dict__)
        if configs.experiment_name[-1] == '_':
            tag = configs.experiment_name[:-1]
        else:
            tag = configs.experiment_name
        experiment.add_tag(tag)
    else:
        experiment = None

    return experiment


def superscale_image(image, f = 1):
    f = 2
    hi, wi = image.shape
    ny = hi // 2
    nx = wi // 2
    tmp = np.reshape(image, (ny, f, nx, f))
    tmp = np.repeat(tmp, f, axis=1)
    tmp = np.repeat(tmp, f, axis=3)
    tmp = np.reshape(tmp, (hi * f, wi * f))

    return tmp


def log_generation_stats(configs, epoch, experiment, sample, agreements, output_analysis):
    if configs.comet:
        for i in range(len(sample)):
            experiment.log_image(np.rot90(sample[i, 0]), name='epoch_{}_sample_{}'.format(epoch, i), image_scale=4, image_colormap='hot')
        experiment.log_metrics(agreements, epoch=epoch)

def log_input_stats(configs, input_analysis, tb_writer=None):
    if tb_writer is not None:
        for i in range(len(input_analysis['training samples'])):
            # experiment.log_image(np.rot90(input_analysis['training samples'][i]), name = 'training example {}'.format(i), image_scale=4, image_colormap='hot')
            img_tensor = torch.Tensor(np.rot90(input_analysis['training samples'][i][0]))
            tb_writer.add_image("train_in", img_tensor, i)


def standardize(data):
    return (data - np.mean(data)) / np.sqrt(np.var(data))


def setup_logger(filename):
    import logging
    logging.basicConfig(
        filename=filename,
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    return logger


def get_path_to_checkpoint(path_to_job, epoch, task=""):
    """
    Get the full path to a checkpoint file.
    Args:
        path_to_job (string): the path to the folder of the current job.
        epoch (int): the number of epoch for the checkpoint.
    """
    if task != "":
        name = "{}_checkpoint_epoch_{:05d}.pyth".format(task, epoch)
    else:
        name = "checkpoint_epoch_{:05d}.pyth".format(epoch)
    return os.path.join(path_to_job, name)


def save_checkpoint(path_to_job, model, optimizer, epoch, cfg):
    """
    Save a checkpoint.
    Args:
        model (model): model to save the weight to the checkpoint.
        optimizer (optim): optimizer to save the historical state.
        epoch (int): current number of epoch of the model.
        cfg (CfgNode): configs to save.
    """
    # Save checkpoints only from the master process.
    if not du.is_master_proc(cfg.NUM_GPUS):
        return
    # Ensure that the checkpoint dir exists.
    os.makedirs(path_to_job, exist_ok=True)
    # Omit the DDP wrapper in the multi-gpu setting.
    sd = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()

    # Record the state.
    checkpoint = {
        "epoch": epoch,
        "model_state": sd,
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg.dump(),
    }
    # Write the checkpoint.
    path_to_checkpoint = get_path_to_checkpoint(
        path_to_job, epoch + 1, cfg.TASK
    )
    with open(path_to_checkpoint, "wb") as f:
        torch.save(checkpoint, f)
    return path_to_checkpoint
