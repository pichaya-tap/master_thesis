import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple

from torch.utils.data import Subset
import glob
from scipy.ndimage import rotate

def extract_centered_subsample(data, z, y, x):
    half_z = 8  # half of 16
    half_y = 8  # half of 16
    half_x = 128# half of 256
    sample = data[z-half_z:z+half_z, y-half_y:y+half_y, x-half_x:x+half_x]
    return sample

def extract_from_filename(filename, param):
    """Extracts energy value from the given filename."""
    parts = filename.split('_')
    if param == 'y':
        # Extract the value
        value_str = parts[2][:-2]

        # Convert '0.1' to 0, otherwise convert to float
        if value_str == '0.1':
            return 0
        else:
            return int(value_str)
        
    if param == 'z':
        return float(parts[3][:-6])
    if param == 'energy':
        return parts[1]
    else:
        return None

def find_watersim_in_folder(energy, folder):
    """Returns the path of the corresponding water simulation file based on the energy value."""
    target_filename = f"DATASET4_{energy}.npy" #which DATASET is the water simulation based on?
    return os.path.join(folder, target_filename)

def find_density_in_folder(dataset,y,z, folder):
    """Returns the path of the corresponding density file based on the dataset and source position value."""
    target_filename = f"{dataset}_{y}Mm_{z}Mm.npy"
    return os.path.join(folder, target_filename)



# Write a custom dataset class (inherits from torch.utils.data.Dataset)

class CustomDataset(Dataset):
    def rotate_sample(self, sample, angle):
        """
        Rotates the given sample by the specified angle around the specified axes.
        :param sample: 3D numpy array
        :param angle: Rotation angle in degrees
        :param axes: A tuple of two axes around which to rotate the sample
        :return: Rotated 3D numpy array
        """
        return rotate(sample, angle, axes= (0,1), reshape=False, mode='nearest')

    def __init__(self, data_folder, density_folder,water_folder, normalization='minmax',augment=False):
        #x_range = [60, 188]
        self.augment = augment 
        self.data_samples = []
        self.density_samples = []

        self.data_names = []
        self.water_samples = []

        # Containers for augmented data
        self.augmented_data_samples = []
        self.augmented_water_samples = []
        self.augmented_density_samples = []

        data_files = glob.glob(os.path.join(data_folder, "*.npy"))           
        print(data_folder)
        # Extract samples first without normalization
        for data_file in data_files:
            data = np.load(data_file)

            # Check if the data array is empty or has any zero dimension
            if data.size == 0 or 0 in data.shape:
                print(f"Invalid data file (empty or zero dimension): {data_file}")
                continue  # Skip the rest of the loop and go to the next file

            # Check if all values in the array are zero
            if np.all(data == 0):
                print(f"All values are zero in file: {data_file}")
                continue  # Skip the rest of the loop and go to the next file

            # Water simulation
            filename = os.path.basename(data_file) # Example "Data1_1500MeV_0Mm_-121.5Mm.npy"
            energy = extract_from_filename(filename, 'energy')
            watersim_path = find_watersim_in_folder(energy, water_folder)
            if not os.path.exists(watersim_path):                
                print(f"Water simulation file not found for {energy}")
                continue
            watersim = np.load(watersim_path) #ratio of primary particles or histories (Water simulation n=10^8 and Phantom simulation n=10^7)
            # watersim shape zyx 160*283*283 or 160*301*301 We want to crop to 16*16*256
            max_index = np.unravel_index(watersim.argmax(), watersim.shape)
            water_sample= extract_centered_subsample(watersim, int(max_index[0]), int(max_index[1]), int(watersim.shape[2]/2)) 

            # Densities
            dataset = filename.split('_')[0] # Example Data1, Data2
            y = extract_from_filename(filename, 'y')
            z = extract_from_filename(filename, 'z')  # Example y=0 and z=-121.5
            density = find_density_in_folder(dataset,y,z, density_folder)
            density_sample = np.load(density)
            
            self.data_samples.append(data)
            self.water_samples.append(water_sample)  
            self.density_samples.append(density_sample)
            self.data_names.append(filename)
            
            if self.augment:
                # Apply augmentation logic here: rotation
                for angle in [90,180]:
                    rotated_data = self.rotate_sample(data, angle)
                    rotated_water = self.rotate_sample(water_sample, angle)
                    rotated_density = self.rotate_sample(density_sample, angle)
                    self.data_samples.append(rotated_data)
                    self.water_samples.append(rotated_water)  
                    self.density_samples.append(rotated_density)
                    self.data_names.append(filename+'_rotated_{}_deg'.format(angle))
           
        # Thresholding
        dose_threshold= 0.1# Set a threshold for values consider close to 0
        density_threshold = 1.5 # Set a threshold to limit outliers
        self.data_samples = [np.where(np.abs(sample) < dose_threshold, 0, sample) for sample in self.data_samples]
        #self.water_samples = [np.where(np.abs(sample) < dose_threshold, 0, sample) for sample in self.water_samples]
        self.density_samples = [np.where(sample > density_threshold, density_threshold, sample) 
                        for sample in self.density_samples]
        
        # Normalize 
        all_data = np.array(self.data_samples)
        print(all_data.shape)
        all_density = np.array(self.density_samples)
        #all_water = np.array(self.water_samples)
        if normalization == 'minmax':
            data_min = 0
            data_range = np.max(all_data) - data_min
            #self.data_samples = [(sample - data_min) / data_range for sample in self.data_samples] # Use global max
            self.data_samples = [sample/ np.max(sample) for sample in self.data_samples] #Use sample max
  
            density_min = np.min(all_density)
            density_range = np.max(all_density) - density_min
            self.density_samples = [(sample - density_min) / density_range for sample in self.density_samples]
            
            #Use max value of each of the samples
            self.water_samples = [sample / np.max(sample) for sample in self.water_samples]

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        data_tensor = torch.tensor(self.data_samples[idx], dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        density_tensor = torch.tensor(self.density_samples[idx], dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        data_name = self.data_names[idx]
        water_tensor = torch.tensor(self.water_samples[idx], dtype=torch.float32).unsqueeze(0) 
        # Concatenate along the channel dimension
        condition = torch.cat([water_tensor, density_tensor], dim=0)
        
        return data_tensor, condition, water_tensor, density_tensor, data_name




