import torch
import torch.nn as nn
import time
import pickle 
import numpy as np
from pathlib import Path
from datetime import datetime
from model3 import Generator
from utils import cal_passing_rate, cal_epoch_rate
from utils import plot_slice, plot_data, cal_delta, mean_absolute_error, cal_delta_average
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import torch.nn.functional as F

# To check the neighboring voxels
def plot_data_matrix(batch_data, batch_fake, data_name, save_folder_path, epoch=1):
    
    voxel = np.arange(256)
    voxel_size = 0.666016*2
    x_vals = (voxel * voxel_size - 170.5)

    # Define the mm values for the ticks within the range of x_vals
    min_x, max_x = min(x_vals), max(x_vals)
    tick_mm_values = np.arange(-150, 150, 25)
    tick_mm_values = tick_mm_values[(tick_mm_values >= min_x) & (tick_mm_values <= max_x)]

    print('batch size:', len(batch_data))
    for i in range(len(batch_data)):
        fig = plt.figure(figsize=(15, 15))  # Adjust figure size for 9 subplots
        gs = gridspec.GridSpec(3, 3)  # 3 rows, 3 columns

        real_dose = batch_data[i].squeeze().cpu().numpy()
        fake_dose = batch_fake[i].squeeze().cpu().numpy()
        max_index = np.unravel_index(real_dose.argmax(), real_dose.shape)

        # Indices for slices
        slice_adjust = [-1, 0, 1]

        # Loop over all combinations of indices
        for j, (dx, dy) in enumerate(itertools.product(slice_adjust, repeat=2)):
            ax = plt.subplot(gs[j])

            # Calculate slice indices
            slice_x = max_index[0] + dx
            slice_y = max_index[1] + dy

            # Extract slices
            real_dose_slice = real_dose[slice_x, slice_y, :]
            fake_dose_slice = fake_dose[slice_x, slice_y, :]

            # Plot real and fake data
            ax.plot(x_vals, real_dose_slice, label='Simulated')
            ax.plot(x_vals, fake_dose_slice, label='Generated', linestyle='--')
            ax.set_ylabel('Dose/Dose_max')
            ax.set_ylim(0,1.0)
            ax.set_xlabel('Depth [mm]')
            ax.set_xticks(tick_mm_values)
            ax.set_xticklabels([f"{mm:.0f}" for mm in tick_mm_values])
            ax.set_title(f'{data_name[i]} - Slice {slice_x+1}, {slice_y+1}')
            ax.legend(loc='upper right')

        plt.suptitle('Comparison of Simulated and Generated Data Dose Along the Beam')
        individual_save_filename = f'epoch{epoch}_{data_name[i][:-4]}.png'
        plt.savefig(os.path.join(save_folder_path, individual_save_filename))
        plt.close(fig)


def test_model(model_g_path, device, test_subset, test_dir):
    
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=8)
    model = Generator()
    # Load the model
    #if device == 'cpu':
    if torch.cuda.is_available() == False:
        state_dict = torch.load(model_g_path, map_location=torch.device('cpu')) # to edit when use gpu
        model.load_state_dict(state_dict)
    else:
        model = model.to(device)
        state_dict = torch.load(model_g_path)
        model.load_state_dict(state_dict)

    model.eval()

    total_mse_loss = 0

    total_delta = []
    total_samples = 0
    startTime = time.time()
    passing_rates1 =[]
    passing_rates3 =[]
    total_mae =[]

    results_data = []
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop over the validation set
        for batch_idx, (real,cond, water_tensor, density_tensor, data_name) in enumerate(test_loader): 
            #if batch_idx ==0:
                cur_batch_size = real.shape[0]    
                # print(f"Processing val batch {batch_idx}")
                # send the input to the device
                real = real.to(device) 
                cond = cond.to(device) 
    
                # Forward pass
                fake = model(cond).float().to(device)

                batch_delta = cal_delta_average(real.detach().cpu(), fake.detach().cpu())
                total_delta += batch_delta
                total_samples += real.size(0)

                batch_maes = mean_absolute_error(real.detach(), fake.detach())
                total_mae += batch_maes
                ################### Visualization ######################

                #with torch.no_grad():
                    #plot_data(real.detach().cpu(), fake.detach().cpu(), water_tensor.detach().cpu(), density_tensor.detach().cpu(), data_name, test_dir, 0 )
                    #plot_slice(real.detach().cpu(), fake.detach().cpu(), density_tensor.detach().cpu() ,data_name, test_dir, 0)
                
                #################### Performance metric ######################
                batch_passing_rates1 = cal_passing_rate(0.01, real.detach(), fake.detach())
                batch_passing_rates3 = cal_passing_rate(0.03, real.detach(), fake.detach())

                # Keep track of the passing_rate
                passing_rates1 += batch_passing_rates1
                passing_rates3 += batch_passing_rates3

                        
                # Compute the sum of MSE loss (per sample)

                mse_loss = F.mse_loss(fake, real,  reduction='sum')

                # Calculate average MSE per voxel for each position (y, z) 
                mse_values = torch.mean((fake - real)**2, dim=(2, 3,4)).cpu().numpy()
                for i in range(cur_batch_size):
                    results_data.append(np.mean(mse_values[i])) #len(results_data) = the number of samples


                # Accumulate the loss
                total_mse_loss += mse_loss.item()

            
        # Compute the average MSE loss over the validation set
        average_delta = sum(total_delta) / len(test_loader.dataset)
        average_mse_loss = total_mse_loss / len(test_loader.dataset)
        average_mse_loss_per_voxel = sum(results_data)/len(results_data)
        average_passing_rates1, average_passing_rates1_error = cal_epoch_rate(passing_rates1)
        average_passing_rates3, average_passing_rates3_error = cal_epoch_rate(passing_rates3)
        average_mse_loss_epoch, mse_error = cal_epoch_rate(results_data)
        average_mae, mae_error =  cal_epoch_rate(total_mae)
            
    print(f"average_delta % at beam center: {average_delta:.2f}")
    print(f"average_mse_loss per sample: {average_mse_loss:.7f}")
    print("average_mse_loss_per_voxel ",average_mse_loss_per_voxel)
    print(f"average_passing_rates1: {average_passing_rates1:.4f} ± {average_passing_rates1_error:.4f}")
    print(f"average_passing_rates3: {average_passing_rates3:.4f} ± {average_passing_rates3_error:.4f}")
    print(f"average_mse_loss_epoch per voxel: {average_mse_loss_epoch:.8f} ± {mse_error:.8f}")
    print(f"average_mae_loss_epoch per voxel: {average_mae:.8f} ± {mae_error:.8f}")


    endTime = time.time()
    print(f"time to predict per sample: {(endTime - startTime)/len(test_subset)}")




if __name__ == "__main__":
    # Define or load model_g_path, device, test_loader, test_dir
     #"/home/tappay01/test/runs/23010349_LRG_0.0009_LRC_8e-06_L_10_opt_RMSprop/MSE_0.925752607848355_generator_epoch945.pth"
    #"/home/tappay01/test/runs/24011735_LRG_0.005_LRC_8e-06_L_10_opt_RMSprop/MSE_1.0923335622045285_generator_epoch975.pth"
    #"/home/tappay01/test/runs/23011327_LRG_0.003_LRC_9e-06_L_10_opt_RMSprop/MSE_0.9634644656001773_generator_epoch963.pth"
    #"/home/tappay01/test/runs/20012319_LRG_0.003_LRC_3e-05_L_10_opt_RMSprop/MSE_0.9230388497707734_generator_epoch943.pth"
    #"/home/tappay01/test/runs/19012243_LRG_0.003_LRC_1e-05_L_10_opt_RMSprop/MSE_0.9618009224097599_generator_epoch919.pth"
    #"/home/tappay01/test/runs/22011337_LRG_0.003_LRC_6e-06_L_10_opt_RMSprop/MSE_1.1231845193328218_generator_epoch950.pth"
    # "/home/tappay01/test/runs/18012253_LRG_0.003_LRC_8e-06_L_10_opt_RMSprop/MSE_0.9358856009638958_generator_epoch977.pth"
    #model_g_path = "/home/tappay01/test/runs/25012241_LRG_0.003_LRC_8e-06_L_10_opt_RMSprop/generator_epoch999.pth"# retrained model
    model_g_path = "/home/tappay01/test/runs/25012241_LRG_0.003_LRC_8e-06_L_10_opt_RMSprop/generator.pth"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    saved_test_dataset ="/home/tappay01/new_data/16011153/test_dataset8.pkl"
    with open(saved_test_dataset, 'rb') as file:
        test_subset = pickle.load(file)
    print('Number of test samples :', len(test_subset))
    
    # Create target directory
    log_dir = "/home/tappay01/test/runs/test_02022024_1200epoch"
    test_dir = Path(log_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    test_model(model_g_path, device , test_subset, test_dir)
