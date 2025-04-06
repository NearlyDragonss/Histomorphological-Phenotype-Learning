from data_manipulation.utils import inception_feature_labels
import numpy as np
import random
import h5py
from torch.utils.data import Dataset
from models.data_augmentation import *
import os
from torchvision.io import decode_image


class DatasetPyTorch(Dataset):
    def __init__(self, hdf5_path, patch_h, patch_w, n_channels, batch_size, thresholds=(), labels=None, empty=False, num_clusters=500, clust_percent=1.0):
        self.i = 0
        self.batch_size = batch_size
        self.done = False
        self.thresholds = thresholds
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.n_channels = n_channels

        # Options for conditional PathologyGAN
        self.num_clusters = num_clusters
        self.clust_percent = clust_percent

        self.labels_name = labels
        if labels is None:
            self.labels_flag = False
        else:
            self.labels_flag = True

        self.hdf5_path = hdf5_path
        self.hdf5_file = None
        if empty:
            self.image_key = None
            self.labels_key = None
            self.size = 0
        else:
            self.image_key, self.labels_key, self.size = self.get_hdf5_keys()
    
    def _lazy_init(self):
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r')


    def get_hdf5_keys(self):
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            labels_name = self.labels_name
            naming = list(hdf5_file.keys())
            if 'images' in naming:
                image_name = 'images'
                if labels_name is None:
                    labels_name = 'labels'       
            else:
                for naming in list(hdf5_file.keys()):     
                    if 'img' in naming or 'image' in naming:
                        image_name = naming
                    elif 'labels' in naming and self.labels_name is None:
                        labels_name = naming
            size = hdf5_file[image_name].shape[0]
        return image_name, labels_name, size


    def __len__(self):

        return self.size # check is right

    def __getitem__(self, idx):
        self._lazy_init()
        batch_images = self.hdf5_file[self.image_key][idx] / 255.0
        batch_labels = self.hdf5_file[self.labels_key][idx]
        return batch_images, batch_labels