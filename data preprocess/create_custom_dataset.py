import torch
print(torch.__version__)
import os
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt 
from datetime import datetime
from pathlib import Path
import time
from tqdm import tqdm
import pickle
import gc
import shutil
import random
from glob import glob

from dataloader import CustomDataset
from model3 import Critic3d, Generator, initialize_weights
from engine import train_step, val_step
from utils import update, generate_report, save, cal_epoch_rate
from torch.utils.data import DataLoader, ConcatDataset, random_split
from test_model import test_model

BATCH_SIZE = 32 
density_folder = "/home/tappay01/new_data/densities_to_dataset4/"
water_folder = "/home/tappay01/new_data/watersimulation/"
'''
data_folders = [
    "/home/tappay01/new_data/resample_to_data4/data1/",
    "/home/tappay01/new_data/resample_to_data4/data2/", 
    "/home/tappay01/new_data/resample_to_data4/data4/",
    "/home/tappay01/new_data/resample_to_data4/data5/",
    "/home/tappay01/new_data/resample_to_data4/data7/"
   
]



# Destination folders
timestamp = datetime.now().strftime("%d%m%H%M")
dataset_dir = "/home/tappay01/new_data/{}/".format(timestamp)
train_folder = dataset_dir + "train"
val_folder = dataset_dir + "validation"
test_folder = dataset_dir + "test"

# Split ratios
train_ratio = 0.7
val_ratio = 0.15

# Gather all file paths
all_files = []
for folder in data_folders:
    files = glob(os.path.join(folder, '*'))  # Adjust pattern as needed
    all_files.extend(files)

# Shuffle files
random.shuffle(all_files)

# Calculate split indices
total_files = len(all_files)
train_index = int(total_files * train_ratio)
val_index = train_index + int(total_files * val_ratio)

# Split files
train_files = all_files[:train_index]
val_files = all_files[train_index:val_index]
test_files = all_files[val_index:]

# Function to copy files
def copy_files(files, destination):
    if not os.path.exists(destination):
        os.makedirs(destination)
        for file in files:
            shutil.copy(file, destination)

# Copy files to respective folders
copy_files(train_files, train_folder)
copy_files(val_files, val_folder)
copy_files(test_files, test_folder)
'''
dataset_dir = "/home/tappay01/new_data/16011153/"
train_folder = dataset_dir + "train"
val_folder = dataset_dir + "validation"
test_folder = dataset_dir + "test"

train_dataset = dataset_dir + 'train_dataset_aug.pkl'
validation_dataset =  dataset_dir + 'val_dataset.pkl'
test_dataset = dataset_dir + 'test_dataset.pkl'
test_dataset8 = dataset_dir + 'test_dataset8.pkl'

# Create datasets for each folder and combine them into a single dataset
train_subset = CustomDataset(train_folder, density_folder, water_folder, augment=True)
save(train_subset, train_dataset)
val_subset = CustomDataset(val_folder, density_folder, water_folder, augment=False)
save(val_subset,validation_dataset )
test_subset = CustomDataset(test_folder, density_folder, water_folder, augment=False)
save(test_subset, test_dataset)
test_subset_dataset8 = CustomDataset("/home/tappay01/new_data/resample_to_data4/data8/", density_folder, water_folder, augment=False)
save(test_subset_dataset8, test_dataset8)



