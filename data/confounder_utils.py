import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from torch.utils.data import Dataset, Subset
from data.celebA_dataset import CelebADataset
from data.cub_dataset import CUBDataset
from data.covid_dataset import CovidDataset
from data.dro_dataset import DRODataset
from data.multinli_dataset import MultiNLIDataset
from data.gdroskin import SkinDataset
from data.cifar10 import CifarDataset

################
### SETTINGS ###
################

confounder_settings = {
    'CelebA':{
        'constructor': CelebADataset
    },
    'CUB':{
        'constructor': CUBDataset
    },
    'MultiNLI':{
        'constructor': MultiNLIDataset
    },
    'Covid':{
        'constructor': CovidDataset
    },
    'Skin':{
        'constructor': SkinDataset
    },
    'Cifar':{
'constructor': CifarDataset
    }

}

########################
### DATA PREPARATION ###
########################
def prepare_confounder_data(args, train, return_full_dataset=False):
    full_dataset = confounder_settings[args.dataset]['constructor'](
        root_dir=args.root_dir,
        target_name=args.target_name,
        confounder_names=args.confounder_names,
        model_type=args.model,
        augment_data=args.augment_data)
    if return_full_dataset:
        y,g=full_dataset.get_groups()
        return DRODataset(
            full_dataset,
            y,
            g,
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str)
    if train:
        splits = ['train', 'val', 'test']
    else:
        splits = ['test']
    subsets,y_subset,g_subset = full_dataset.get_splits(splits, train_frac=args.fraction)
    dro_subsets = [DRODataset(subsets[split], y_subset[split],g_subset[split],process_item_fn=None, n_groups=full_dataset.n_groups,
                              n_classes=full_dataset.n_classes, group_str_fn=full_dataset.group_str) \
                   for split in splits]
    return dro_subsets
