import torch
from yacs.config import CfgNode as CN


_C = CN()

_C.comet = False
_C.init_method = "env://"
_C.CUDA = True if torch.cuda.is_available() else False
_C.NUM_GPUS = 1
_C.ENABLE_AMP = False

# Data
_C.training_dataset = "3d-water"
_C.dataset_seed = 0
_C.classes = 3 # the water dataset has 3 classes: empty, O, H
_C.condition_material_dim = 1

# Train
_C.RNG_SEED = 0
_C.lr = 1e-2
_C.max_epochs = 100 #500
_C.ckpt_save_period = 10 # save model every n epochs
_C.batch_size = 16
_C.convergence_moving_average_window = 5 # not sure what value it should be
_C.convergence_margin = 0.1

# Model
_C.model = "gated1"
_C.activation_function = "relu"
_C.conv_size = 3
_C.conv_layers = 20
_C.conv_filters = 20
_C.fc_dropout_probability = 0.21

# Generation
_C.sample_generation_mode = "serial"
_C.sample_outpaint_ratio = 2 # number of times to extrapolate
_C.boundary_layers = 1
_C.n_samples = 1 # total number of samples to generate
_C.sample_batch_size = 1 # number of samples to generate at one time

# Output
_C.OUTPUT_DIR = "checkpoints/water-default"