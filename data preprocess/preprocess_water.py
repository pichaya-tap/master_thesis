# xzy
datasets = {
    "Data1": {
        "voxel_dim": [0.707031*2, 0.707031*2, 1.25*2],
        "origin": [-181, -181, -229],
        "positions_x": [-110],
        "positions_y": [-50, -25, 0, 25, 50, 75],
        "positions_z": [-146.5, -121.5, -96.5, -71.5, -46.5],
        "voxel_center": [0,0,-71.5]
    },
    "Data2": {
        "voxel_dim": [0.759766*2, 0.759766*2, 1.25*2],
        "origin": [-194.5, -194.5, -271],
        "positions_x": [-110],
        "positions_y": [-25, 0, 25, 50, 75],
        "positions_z": [-171.5, -146.5, -121.5, -96.5, -71.5],
        "voxel_center": [0,0,-72.25]
    },
    "Data4": {
        "voxel_dim": [0.666016*2, 0.666016*2, 1.25*2],
        "origin": [-170.5,-170.5, -188.75],
        "positions_x": [-110],
        "positions_y": [-50, -25, 0, 25, 50, 75],
        "positions_z": [-71.5, -46.5, -21.5, 3.5],
        "voxel_center": [0,0,-32.5]
    },
    "Data5": {
        "voxel_dim": [0.689453*2, 0.689453*2, 1.25*2],
        "origin": [-176.5, -176.5, -208.25],
        "positions_x": [-110],
        "positions_y": [-50, -25, 0, 25, 50, 75],
        "positions_z": [-96.5, -71.5, -46.5, -21.5],
        "voxel_center": [0,0,-44.5]
    },
    "Data7": {
        "voxel_dim": [0.683594*2, 0.683594*2, 1.25*2],
        "origin": [-175, -175, -199.25],
        "positions_x": [-110],
        "positions_y": [-25, 0, 25, 50, 75],
        "positions_z": [-96.5, -71.5, -46.5, -21.5],
        "voxel_center": [0,0,-29.25]
    }
}

import SimpleITK as sitk
import os
import numpy as np
import glob


def set_sitk_image(array, dataset):
    # Convert numpy array to SimpleITK image
    sitk_image = sitk.GetImageFromArray(array)  
    # Set the spacing for data
    spacing = datasets[dataset]["voxel_dim"]
    sitk_image.SetSpacing(spacing) #xyz
    origin = datasets[dataset]["origin"]
    sitk_image.SetOrigin(origin)
    print(sitk_image.GetSpacing(),sitk_image.GetOrigin() )
    return sitk_image

    
def resample_image(input_image, new_spacing, new_origin):
    size = input_image.GetSize()
    spacing = input_image.GetSpacing()
    new_size = [int(size[d] * spacing[d] / new_spacing[d]) for d in range(3)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(new_origin)
    resampler.SetOutputDirection(input_image.GetDirection())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor) # preserve the original data values without smoothing
    
    resampled_image = resampler.Execute(input_image)
    return resampled_image

def extract_from_filename(filename, param):
    """Extracts energy value from the given filename."""
    parts = filename.split('_')
    if param == 'y':
        return float(parts[2][:-2])
    if param == 'z':
        return float(parts[3][:-6])
    if param == 'energy':
        return parts[1]
    else:
        return None

# function to extract ROI and convert to numpy
def extract_roi_array(image, roi_size, center_physical):
    source_index = image.TransformPhysicalPointToIndex(center_physical)
    start_index = [int(source_index[0]-roi_size[0]*0.5), int(source_index[1]-roi_size[1]*0.5), int(source_index[2]-roi_size[2]*0.5)]
    roi = sitk.RegionOfInterest(image, roi_size, start_index)
    return sitk.GetArrayFromImage(roi) 


dataset = '4'#Current

data_folder = '/scratch/tappay01/watersimulation/DATASET4/'
data_files = glob.glob(os.path.join(data_folder, "*.npy"))  
save_folder = '/scratch/tappay01/watersimulation/resampled_DATASET4/'
# target = data1
target_spacing = datasets["Data1"]["voxel_dim"]
target_origin = datasets["Data1"]["origin"] # Do not change the origin!!
roi_size = (240,16,16) #xyz  
# center_physical = datasets["data1"]["voxel_center"] for full size
 


for file_path in data_files:   
    # Extract the base file name from the full path
    filename = os.path.basename(file_path)
    save_path = os.path.join(save_folder, filename)
    # check if the target file exists already, skip
    if os.path.exists(save_path):
        print(f"Warning: File '{save_path}' already exists. Skipping calculation.")
        continue

    # Load data from .npy
    array = np.load(file_path)
    # Convert numpy array to SimpleITK image and set parameters
    image = set_sitk_image(array, 'Data'+dataset)

    if dataset != '1':
        # Resample the image
        resampled = resample_image(image, target_spacing, target_origin)
    if dataset == '1':
        resampled = image
    # range 50 - 241 voxel source= 291/2= 145

    source_physical = [24.0, 0, 0]
    roi_array = extract_roi_array(resampled, roi_size, source_physical)
    print(roi_array.shape)# (16, 16, 240)
    #np.save(save_path, roi_array)
