""" train and test dataset

author baiyu
"""
import os
import sys
import pickle

from skimage import io
import matplotlib.pyplot as plt
import numpy
import torch
from torch.utils.data import Dataset

class CIFAR100Split(Dataset):
    """CIFAR-100 dataset split by classes for continual learning."""

    def __init__(self, path, class_indices, transform=None):
        """
        Args:
            path (str): Path to the dataset.
            class_indices (list): List of class indices to include in this split.
            transform (callable, optional): Transform to apply to the images.
        """
        with open(os.path.join(path, 'train'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')

        self.transform = transform

        # Filter data by class indices
        self.filtered_indices = [
            i for i, label in enumerate(self.data['fine_labels'.encode()])
            if label in class_indices
        ]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, index):
        actual_index = self.filtered_indices[index]
        label = self.data['fine_labels'.encode()][actual_index]
        r = self.data['data'.encode()][actual_index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][actual_index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][actual_index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return label, image
    
class CIFAR100Test(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        with open(os.path.join(path, 'test'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.data['data'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return label, image

