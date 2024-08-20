import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from torch.utils.data import Dataset, Subset
from data.confounder_dataset import ConfounderDataset
import pickle
class CifarDataset(ConfounderDataset):
    """
    CUB dataset (already cropped and centered).
    Note: metadata_df is one-indexed.
    """

    def __init__(self, root_dir,
                 target_name, confounder_names,
                 augment_data=False,
                 model_type=None):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.model_type = model_type
        self.augment_data = augment_data

        self.data_dir=''
        cifar10_dir = os.path.join('/data' 'cifar-10-batches-py')
        train_data = []
        train_labels = []
        for batch in range(1, 6):
            with open(os.path.join(cifar10_dir, f'data_batch_{batch}'), 'rb') as f:
                batch_data = pickle.load(f, encoding='bytes')
                train_data.append(batch_data[b'data'])
                train_labels.append(batch_data[b'labels'])
        train_data = np.concatenate(train_data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        train_labels = np.concatenate(train_labels)
        with open(os.path.join(cifar10_dir, 'test_batch'), 'rb') as f:
            test_data1 = pickle.load(f, encoding='bytes')
            test_data = test_data1[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            test_labels = np.array(test_data1[b'labels'])
        # Read in metadata
        val_data=train_data[40000:,:,:,:]
        val_label=train_labels[40000:]
        #train_data=train_data[:40000, :, :, :]
        #train_labels = train_labels[:40000]
        num1=train_data.shape[0]
        num2=val_data.shape[0]
        num3=test_data.shape[0]


        # Get the y values
        self.y_array = np.concatenate((train_labels,val_label,test_labels))
        self.n_classes = 10

        # We only support one confounder for CUB for now
        #self.confounder_array = self.metadata_df['group'].values
        #self.n_confounders = 1
        # Map to groups
        self.n_groups = 10
        self.group_array = self.y_array

        # Extract filenames and splits
        self.filename_array = np.concatenate((train_data,val_data,test_data))
        self.n_confounders = 1
        self.split_array = np.array([0]*num1+[1]*num2+[2]*num3)
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        # Set transform
        if model_attributes[self.model_type]['feature_type']=='precomputed':
            self.features_mat = torch.from_numpy(np.load(
                os.path.join(root_dir, 'features', model_attributes[self.model_type]['feature_filename']))).float()
            self.train_transform = None
            self.eval_transform = None
        else:
            self.features_mat = None
            self.train_transform = get_transform_skin(
                self.model_type,
                train=True,
                augment_data=augment_data)
            self.eval_transform = get_transform_skin(
                self.model_type,
                train=False,
                augment_data=augment_data)

    def get_groups(self):
        return self.y_array,self.group_array


    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        x=self.train_transform(Image.fromarray(np.uint8(self.filename_array[idx])))

        return x, y, g, idx

def get_transform_skin(model_type, train, augment_data):
    # scale = 256.0/224.0
    #target_resolution = model_attributes[model_type]['target_resolution']
    #assert target_resolution is not None

    if (not train) or (not augment_data):
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose([

            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),

        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    return transform
