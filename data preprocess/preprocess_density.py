from preprocess import set_sitk_image, resample_image, extract_roi_array
import os
import numpy as np



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
        "positions_z": [ -121.5, -71.5, -96.5,-146.5], #exclude -171.5,-71.5, -96.5,-146.5
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
    },
    "Data8": {
        "voxel_dim": [0.726562*2, 0.726562*2, 1.25*2],
        "origin": [-186, -186, -186.25],
        "positions_x": [-110],
        "positions_y": [-25, 0, 25, 50],
        "positions_z": [-96.5, -71.5, -46.5, -21.5, 3.5],
        "voxel_center": [0,0,-16.25]
    }
}

def resample_density(density_path, target_dataset):
    '''Take input as density file path and convert to SimpleITK image and resample to physical space of Data1
return resampled SimpleITK image'''

    filename = os.path.basename(density_path) # Example "Data2.npy"
    dataset = filename.split('.')[0]    #Data2
    target_spacing = datasets[target_dataset]["voxel_dim"]
    target_origin = datasets[target_dataset]["origin"]
    
    #print(dataset)
    # Load data from .npy
    array = np.load(density_path)
    #print(array.shape)
    # Convert numpy array to SimpleITK image and set parameters
    image = set_sitk_image(array, dataset)

    if dataset != target_dataset:
        # Resample the image
        resampled = resample_image(image, target_spacing, target_origin)
    if dataset == target_dataset:
        resampled = image
    return resampled



def create_density_roi(dataset, target_dataset):
    roi_size = (256,16,16) #xyz  
    density_folder =  "/home/tappay01/new_data/densities/"
    original_density_path = os.path.join(density_folder,  dataset +'.npy')
    resampled = resample_density(original_density_path, target_dataset)
    # Extract ROI    
    # y z source position extract from Dataset notes
    # source at x postion = -110
    # full X range size 256
    for y in datasets[dataset]["positions_y"]:
        for z in datasets[dataset]["positions_z"]:
            source_physical = [0, y, z] #Mm
            print('source_physical: ',source_physical)
            density_roi_array = extract_roi_array(resampled, roi_size, source_physical)
            np.save(os.path.join('/home/tappay01/new_data/densities_to_dataset4/','{}_{}Mm_{}Mm.npy'.format(dataset, y, z)), density_roi_array)
'''
"/gpfs001/scratch/schwar14/simudosemapheavyions/DosemapInBrain/Dataset/densities.npy"
"/gpfs001/scratch/schwar14/simudosemapheavyions/DosemapInBrain/Dataset2/densities.npy"
"/gpfs001/scratch/schwar14/simudosemapheavyions/DosemapInBrain/Dataset4/densities.npy"
"/gpfs001/scratch/schwar14/simudosemapheavyions/DosemapInBrain/Dataset5/densities.npy"
"/gpfs001/scratch/schwar14/simudosemapheavyions/DosemapInBrain/Dataset7/densities.npy"
'''
create_density_roi('Data8', 'Data4')

