import pickle
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import Dataset


class build_dataset(Dataset):
    def __init__(self, configs):
        np.random.seed(configs.dataset_seed)
        if configs.training_dataset == '3d-idealgas':
            self.samples = np.load('data/hard_sphere_gas_1000.npy', allow_pickle=True)
        elif configs.training_dataset == '3d-water':
            self.samples = np.load('data/water_648atoms/water.npy', allow_pickle=True) # array(1000, 648, 3)
            self.samples_identity = np.load('data/water_648atoms/water_identity.npy', allow_pickle=True) # array(1000, 648)
        elif configs.training_dataset == 'sb2s3':
            self.samples = np.load('data/sb2s3_3000K_480atoms_5000frames/3000K_positions.npy', allow_pickle=True)
            self.samples_identity = np.load('data/sb2s3_3000K_480atoms_5000frames/3000K_element_symbols.npy', allow_pickle=True)

        self.samples = transform_data_2(self.samples,self.samples_identity)

        self.samples = np.expand_dims(self.samples, axis=1) # (1000, 80, 80, 80)
        self.num_conditioning_variables = self.samples.shape[1] - 1
        assert self.samples.ndim == 5

        ##### Data Augmentation
        self.samples=np.concatenate((self.samples[:,:,0:40,0:40,0:40],self.samples[:,:,40:,40:,40:],self.samples[:,:,40:,0:40,0:40],self.samples[:,:,0:40,40:,0:40],self.samples[:,:,0:40,0:40,40:],self.samples[:,:,0:40,40:,40:],self.samples[:,:,40:,0:40,40:],self.samples[:,:,40:,40:,0:40]))
        rot=np.rot90(self.samples.copy(),k=1,axes=(2,3)) # (8000, 1, 40, 40, 40)
        rot2=np.rot90(self.samples.copy(),k=2,axes=(2,3))
        rot3=np.rot90(self.samples.copy(),k=1,axes=(2,4))
        rot4=np.rot90(self.samples.copy(),k=2,axes=(2,4))
        rot5=np.rot90(self.samples.copy(),k=1,axes=(3,4))
        rot6=np.rot90(self.samples.copy(),k=2,axes=(3,4))
        self.samples = np.concatenate((self.samples, rot,rot2,rot3,rot4,rot5,rot6), axis=0) # (56000, 1, 40, 40, 40)

        np.random.shuffle(self.samples)
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
        a_file = open("datadims.pkl", "wb")
        pickle.dump(self.dataDims, a_file)
        a_file.close()
	
        # normalize pixel inputs
        self.samples[:,0,:,:,:] = np.array((self.samples[:,0] + 1)/(self.dataDims['classes'])) # normalize inputs on 0--1

        torch.cuda.empty_cache()
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    


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


def get_dataloaders(configs):
    dataset = build_dataset(configs)  # get data
    dataDims = dataset.dataDims
    print(['dataset',len(dataset)])
    train_size = int(0.8 * len(dataset))  # split data into training and test sets
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.Subset(dataset, [range(train_size),range(train_size,test_size + train_size)])  # split it the same way every time
    tr = data.DataLoader(train_dataset, batch_size=configs.training_batch_size, shuffle=True, num_workers= 0, pin_memory=True)  # build dataloaders
    te = data.DataLoader(test_dataset, batch_size=configs.training_batch_size, shuffle=False, num_workers= 0, pin_memory=True)
    print([type(tr),len(te)])
    return tr, te, dataDims
