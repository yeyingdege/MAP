import os
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import Dataset


class build_dataset(Dataset):
    def __init__(self, configs, split='train', use_cache=True):
        self.split = split
        self.cfg = configs
        
        if use_cache and os.path.exists(self.cfg.CACHE_FILE):
            self.samples = np.load(self.cfg.CACHE_FILE, allow_pickle=True)
        else:
            self.samples = self.process_raw_data()

        self.num_conditioning_variables = self.samples.shape[1] - 1
        assert self.samples.ndim == 5

        self.dataDims = {
            'classes' : len(np.unique(self.samples)), # 3 classes: 0, 1-H, 2-O
            'input x dim' : self.samples.shape[-1],
            'input y dim' : self.samples.shape[-2],
            'input z dim': self.samples.shape[-3],
            'channels' : 1, # hardcode as one so we don't get confused with conditioning variables
            'dataset length' : len(self.samples),
            'sample x dim' : self.samples.shape[-1] * configs.sample_outpaint_ratio,
            'sample y dim' : self.samples.shape[-2] * configs.sample_outpaint_ratio,
            'sample z dim': self.samples.shape[-3] * configs.sample_outpaint_ratio,
            'num conditioning variables' : self.num_conditioning_variables,
            'conv field' : configs.conv_layers + configs.conv_size // 2,
           # 'conditional mean' : self.conditional_mean,
           # 'conditional std' : self.conditional_std
        }

        # normalize pixel inputs
        self.samples[:,0,:,:,:] = np.array((self.samples[:,0] + 1)/(self.dataDims['classes']), dtype=float) # normalize inputs on 0--1

        train_size = int(0.8 * len(self.samples))  # split data into training and test sets
        # test_size = len(self.samples) - train_size
        if self.split == 'train':
            self.samples = self.samples[:train_size]
        else:
            self.samples = self.samples[train_size:]
        # torch.cuda.empty_cache()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.from_numpy(self.samples[idx].copy())

    def process_raw_data(self):
        if self.cfg.training_dataset == '3d-idealgas':
            samples = np.load('data/hard_sphere_gas_1000.npy', allow_pickle=True)
        elif self.cfg.training_dataset == '3d-water':
            samples = np.load('data/water_648atoms/water.npy', allow_pickle=True) # array(1000, 648, 3)
            samples_identity = np.load('data/water_648atoms/water_identity.npy', allow_pickle=True) # array(1000, 648)
        elif self.cfg.training_dataset == 'sb2s3':
            samples = np.load('data/sb2s3_3000K_480atoms_5000frames/3000K_positions.npy', allow_pickle=True)
            samples_identity = np.load('data/sb2s3_3000K_480atoms_5000frames/3000K_element_symbols.npy', allow_pickle=True)

        samples = transform_data_2(samples, samples_identity)

        samples = np.expand_dims(samples, axis=1) # (1000, 80, 80, 80)
        ##### Data Augmentation
        samples=np.concatenate((samples[:,:,0:40,0:40,0:40],samples[:,:,40:,40:,40:],samples[:,:,40:,0:40,0:40],samples[:,:,0:40,40:,0:40],samples[:,:,0:40,0:40,40:],samples[:,:,0:40,40:,40:],samples[:,:,40:,0:40,40:],samples[:,:,40:,40:,0:40]))
        rot=np.rot90(samples.copy(),k=1,axes=(2,3)) # (8000, 1, 40, 40, 40)
        rot2=np.rot90(samples.copy(),k=2,axes=(2,3))
        rot3=np.rot90(samples.copy(),k=1,axes=(2,4))
        rot4=np.rot90(samples.copy(),k=2,axes=(2,4))
        rot5=np.rot90(samples.copy(),k=1,axes=(3,4))
        rot6=np.rot90(samples.copy(),k=2,axes=(3,4))
        samples = np.concatenate((samples, rot,rot2,rot3,rot4,rot5,rot6), axis=0) # (56000, 1, 40, 40, 40)

        np.random.seed(self.cfg.dataset_seed)
        np.random.shuffle(samples)
        np.save(self.cfg.CACHE_FILE, samples)
        return samples


def transform_data(sample):
    newdata = np.zeros((sample.shape[0], 90, 90, 90))
    for i in range(0, sample.shape[0]):
        for j in range(0, sample.shape[1]):
                #print([i,j,int((sample[i + 1, j, 2])/ 0.08), int((sample[i + 1, j, 1])/ 0.08), int((sample[i + 1, j, 0])/ 0.08)])

                newdata[i, int((sample[i , j, 2])/ 0.2), int((sample[i, j, 1])/ 0.2), int((sample[i , j, 0])/ 0.2)] = 1
    return newdata


def transform_data_2(sample,sample_identity):
    newdata = np.zeros((sample.shape[0], 80, 80, 80))
    for i in range(0, len(sample)):
        for j in range(0, sample.shape[1]):
                #print([i,j,int((sample[i + 1, j, 2])/ 0.08), int((sample[i + 1, j, 1])/ 0.08), int((sample[i + 1, j, 0])/ 0.08)])

                if np.isnan(sample[i,j,:]).any() == False and sum(sample[i,j,:]>100)<1:
                    if sample_identity[i,j] == 'H':
                        newdata[i, int((sample[i , j, 2])/ 0.25), int((sample[i, j, 1])/ 0.25), int((sample[i , j, 0])/ 0.25)] = 1
                    if sample_identity[i,j] == 'O':
                        newdata[i, int((sample[i, j, 2]) / 0.25), int((sample[i, j, 1]) / 0.25), int((sample[i, j, 0]) / 0.25)] = 2
    return newdata


def transform_data_3(sample):
    newdata = np.zeros((sample.shape[0],32,32,32))
    for i in range(0, len(sample)):
        for j in range(0, sample.shape[1]):
                #print([i,j,int((sample[i + 1, j, 2])/ 0.08), int((sample[i + 1, j, 1])/ 0.08), int((sample[i + 1, j, 0])/ 0.08)])

                if np.isnan(sample[i,j,:]).any() == False:


                    newdata[i, int((sample[i , j, 2])/ 0.25), int((sample[i, j, 1])/0.25 ), int((sample[i , j, 0])/0.25)] = 1
    return newdata



def get_dataset(cfg):
    train_dataset = build_dataset(cfg, split='train')
    test_dataset = build_dataset(cfg, split='test')
    dataDims = train_dataset.dataDims
    return train_dataset, test_dataset, dataDims

