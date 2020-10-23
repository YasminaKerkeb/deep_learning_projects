
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset 
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import torchvision.models as models
import os
import seaborn as sns

#We create our own custom Dataset following the example on this link: 
#https://pytorch.org/tutorials/beginner/data_loading_tutorial.html


class StanfordMRDataset(Dataset):

    def __init__(self, train, target, plane, root_dir,transform=None):
            """
            Args:
            ----

                train (boolean):                Boolean variable determining whether to load train or validation data
                target (string):                Task to implement ('acl','meniscus','abnormal')
                plane (string):                'coronal', 'axial' or 'sagittal'
                root_dir (string):              Directory with all the images and data.
                transform (callable, optional): Optional transform to be applied
                                                on a sample.
            """
            super(StanfordMRDataset, self).__init__()
            self.root_dir = root_dir
            self.target = target
            self.plane=plane
            self.transform = transform
            if train:
                train_or_valid = 'train'
            else:
                train_or_valid = 'valid'
            self.filename=self.root_dir+train_or_valid+'-'+self.target+'.csv'
                
            self.data = pd.read_csv(self.filename,header=None,
                       names=['id', 'label'], 
                       dtype={'id': str, 'label': np.int64})
            self.img_folder = self.root_dir+train_or_valid+'/'+self.plane+'/'


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        mri_img = np.load(self.img_folder+idx+'.npy')
        label = self.data.iloc[idx:]['label']
        
        if self.transform:
            mri_img = self.transform(mri_img)

        sample = {'image': mri_img, 'label': label}

        return sample

