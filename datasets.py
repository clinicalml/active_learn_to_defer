import math
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
import argparse
import os
import shutil
import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # THIS IS BAD! but okay for now, should pass device to dataset constructor

class CifarExpertDataset(Dataset):
    def __init__(self, images, targets, expert_fn, labeled, indices = None, expert_preds = None):
        """
        Original cifar dataset
        images: images
        targets: labels
        expert_fn: expert function
        labeled: indicator array if images is labeled
        indices: indices in original CIFAR dataset (if this subset is subsampled)
        expert_preds: used if expert_fn or have different expert model
        """
        self.images = images
        self.targets = np.array(targets)
        self.expert_fn = expert_fn
        self.labeled = np.array(labeled)
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        self.transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        if expert_preds is not None:
            self.expert_preds = expert_preds
        else:
            self.expert_preds = np.array(expert_fn(None, torch.FloatTensor(targets)))
        for i in range(len(self.expert_preds)):
            if self.labeled[i] == 0:
                self.expert_preds[i] = -1 # not labeled by expert
        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.array(list(range(len(self.targets))))
    def __getitem__(self, index):
        """Take the index of item and returns the image, label, expert prediction and index in original dataset"""
        label = self.targets[index]
        image = self.transform_test(self.images[index])
        expert_pred = self.expert_preds[index]
        indice = self.indices[index]
        labeled = self.labeled[index]
        return torch.FloatTensor(image), label, expert_pred, indice, labeled

    def __len__(self):
        return len(self.targets)
    
    
    
    
class CifarExpertDatasetLinear(Dataset):
    def __init__(self, images, targets, expert_fn, labeled,  indices = None, model = None):
        """
        Original cifar dataset
        images: images
        targets: labels
        expert_fn: expert function
        labeled: indicator array if images is labeled
        indices: indices in original CIFAR dataset (if this subset is subsampled)
        model: model that maps images to a vector representation
        """
        self.images = images
        self.targets = np.array(targets)
        self.expert_fn = expert_fn
        self.model = model
        self.labeled = np.array(labeled)
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        self.transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        self.expert_preds = np.array(expert_fn(None, torch.FloatTensor(targets)))
        for i in range(len(self.expert_preds)):
            if self.labeled[i] == 0:
                self.expert_preds[i] = -1 # not labeled by expert
        if indices != None:
            self.indices = indices
        else:
            self.indices = np.array(list(range(len(self.targets))))
    def __getitem__(self, index):
        """Take the index of item and returns the image, label, expert prediction and index in original dataset"""
        label = self.targets[index]
        image = self.transform_test(self.images[index])
        image_repr = image.to(device)
        image_repr = torch.reshape(image_repr, (1,3,32,32)).to(device)
        image_repr = self.model.repr(image_repr)
        image_repr = image_repr[0]
        image_repr = image_repr.to(torch.device('cpu'))
        expert_pred = self.expert_preds[index]
        indice = self.indices[index]
        labeled = self.labeled[index]
        return image_repr, label, expert_pred, indice, torch.FloatTensor(image)

    def __len__(self):
        return len(self.targets)