import os
import torch
from collections import OrderedDict
import util.distributed as du


def get_checkpoint_path(directory):
    names = os.listdir(directory)
    names = [f for f in names if f.startswith("model") and (f.endswith(".pt") or f.endswith(".pth"))]
    return [os.path.join(directory, f) for f in names]


def has_checkpoint(directory):
    names = get_checkpoint_path(directory)
    if len(names) > 0:
        return True, names
    else:
        return False, None


def get_last_checkpoint(directory):
    names = get_checkpoint_path(directory)
    if len(names) == 0:
        return None
    else:
        name = sorted(names)[-1]
        return name


def load_checkpoint(ckpt_path: str, model: object, optimizer=None, scaler=None, is_distributed=False):
    ckpt_ = torch.load(ckpt_path, weights_only=False)
    cfg = ckpt_["cfg"]
    if not is_distributed and 'module' in list(ckpt_["model_state_dict"].keys())[0]:
        new_state_dict = OrderedDict()
        for k, v in ckpt_["model_state_dict"].items():
            name = k[7:] # remove 'module.' of dataparallel
            new_state_dict[name]=v
        model.load_state_dict(new_state_dict)
    elif is_distributed and 'module.' not in list(ckpt_["model_state_dict"].keys())[0]:
        new_state_dict = OrderedDict()
        for k, v in ckpt_["model_state_dict"].items():
            name = 'module.' + k # add 'module.'
            new_state_dict[name]=v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(ckpt_["model_state_dict"])
    # model= model.to("cuda" if cfg.CUDA else "cpu")
    if optimizer is not None:
        optimizer.load_state_dict(ckpt_["optimizer_state_dict"])
    if scaler is not None and "scaler_state" in ckpt_:
        scaler.load_state_dict(ckpt_["scaler_state"])
    return ckpt_["epoch"]


def save_checkpoint(model, optimizer, epoch, cfg, scaler=None):
    if not du.is_master_proc(num_gpus=cfg.NUM_GPUS):
        return
    checkpoint = {
        'model_state_dict': model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'cfg': cfg
    }
    if scaler is not None:
        checkpoint["scaler_state"] = scaler.state_dict()
    torch.save(checkpoint, os.path.join(cfg.OUTPUT_DIR, 'model-ep{:05d}.pt'.format(epoch)))
