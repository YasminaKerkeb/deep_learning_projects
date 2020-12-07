import shutil
import os
import time
from datetime import datetime
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchsample.transforms import RandomRotate, RandomTranslate, RandomFlip, ToTensor, Compose, RandomAffine
from torchvision import transforms
import torch.nn.functional as F
from data_preprocessor import MRDataset
import model

from sklearn import metrics




augmentor = Compose([
    transforms.Lambda(lambda x: torch.Tensor(x)),
    RandomRotate(25),
    RandomTranslate([0.11, 0.11]),
    RandomFlip(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)),
])

#Load Dataset
train_dataset = MRDataset('./data/', True,
                            'axial', transform=augmentor, train=True)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1, shuffle=True, num_workers=11, drop_last=False)

validation_dataset = MRDataset(
    './data/', False, 'axial', train=False)
validation_loader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=1, shuffle=-True, num_workers=11, drop_last=False)

mrnet = model.MRNet()

#Train

mrnet.compute_training(train_loader,validation_loader)


