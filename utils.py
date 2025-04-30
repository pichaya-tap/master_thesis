import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
import matplotlib.gridspec as gridspec
from datetime import datetime
from matplotlib import gridspec
import numpy as np
from scipy import stats
import pickle
from matplotlib.gridspec import GridSpec

def save(data, save_file):
    with open(save_file, 'wb') as file:
        pickle.dump(data, file)
        print("saved to ",save_file)

def generate_report(results):
    report = ""
    '''
    rate_keys = ["train_passing_rate1", "train_passing_rate3", 
                 "val_passing_rate1", "val_passing_rate3"]
    error_keys = ["train_error1", "train_error3", 
                  "val_error1", "val_error3"]

    # Handling rate keys
    for i, rate_key in enumerate(rate_keys):
        max_rate = max(results[rate_key])
        corresponding_error = results[error_keys[i]][results[rate_key].index(max_rate)]
        report += f"{rate_key}: Highest Rate: {max_rate:.1f} (±{corresponding_error:.4f})\n"
    '''
    
    min_mse = min(results["mse"])

    min_val_delta = min(results["val_delta"])
    min_train_delta = min(results["train_delta"])
    max_passing_rate3 = max(results["val_passing_rate3"])
    highest_index = results["val_passing_rate3"].index(max_passing_rate3)
    val_error3 = results["val_error3"][highest_index]

    report += f"\ndelta: Lowest train delta: {min_train_delta:.4f} \n"
    report += f"delta: Lowest validation delta: {min_val_delta:.4f} \n"    
    report += f"mse: Lowest MSE: {min_mse:.4f} \n"
    report += f"mse: Lowest MSE: {min_mse:.4f} \n"
    report += f"passing rate3%: highest : {max_passing_rate3:.2f} +- {val_error3:.2f} \n"

    return report

def unravel_index(index, shape):
    """
    Convert a flat index into an index tuple of coordinates in a tensor of the given shape.

    Args:
        index: A flat index.
        shape: Shape of the tensor to which the index corresponds.

    Returns:
        A tuple of coordinates.
    """
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))

def cal_delta_average(batch_data, batch_fake):
    average_delta_list=[]
    for i in range(len(batch_data)):

        real_dose = batch_data[i].squeeze().cpu().numpy()
        fake_dose = batch_fake[i].squeeze().cpu().numpy()
        max_index = np.unravel_index(real_dose.argmax(), real_dose.shape)

        # Extract slices
        real_dose_slice = real_dose[int(max_index[0]), int(max_index[1]), :] #np.sum(real_dose, axis=(0, 1)) 
        fake_dose_slice = fake_dose[int(max_index[0]), int(max_index[1]), :] #np.sum(fake_dose, axis=(0, 1)) 



        # Plotting the delta percent on the bottom subplot
        max_real_dose = np.max(real_dose_slice)
        delta_percent = 100 * (fake_dose_slice - real_dose_slice) / max_real_dose
        # Calculate the average
        #print(max(abs(delta_percent)))
        average_delta_percent = np.mean(abs(delta_percent))
        average_delta_list.append(average_delta_percent)

    return(average_delta_list)



def cal_delta(real, fake):
    # Vectorized computation of delta values
    deltas = torch.abs(fake - real) * real
    return deltas

def mean_absolute_error(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MAE) between two vectors.

    Parameters:
    - y_true: Ground truth values (actual values).
    - y_pred: Predicted values.

    Returns:
    - mean absolute error.
    """
    # Ensure that both input arrays have the same shape
    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape.")

    # Calculate absolute differences element-wise
    absolute_errors = torch.abs(y_true - y_pred)

    # Calculate mean absolute errors for each item in the batch
    batch_maes = torch.mean(absolute_errors, dim=(1, 2, 3)).tolist()

    return batch_maes


def cal_passing_rate(i, real, fake):
    """
    Calculate weighted passing rate where the beam center is given more importance.

    Args:
        i: Delta percent, e.g., 0.01 or 0.03.
        real: Batch of real dataset on GPU.
        fake: Batch of generated data on GPU.

    Returns:
        A list of passing rates for each sample in the batch.
    """

    batch_passing_rates = []
    # Create a mask for voxels where real is greater than 0.0
    mask = real > 0.0 # to adjust

    for b in range(real.shape[0]):  # Iterate over the batch
        delta = cal_delta(real, fake)
        # Apply the mask to the delta condition
        passing_condition = (delta < i)# & mask

        passing_voxels = torch.sum(passing_condition, dim=(2, 3, 4)).squeeze().tolist()

        # Total weight
        total_voxel =  int(real.shape[2]*real.shape[3]*real.shape[4])
        #non_zero_voxel =  torch.sum(mask).item()
        # Calculate weighted passing rate for the current sample in the batch
        batch_passing_rates = [passing_voxel * 100 / total_voxel for  passing_voxel in passing_voxels]

    return batch_passing_rates



def cal_epoch_rate(rate):
    # Calculate the mean and SEM for the epoch
    mean = np.mean(rate)
    std_dev = np.std(rate)
    sample_size = len(rate)
    sem = std_dev / np.sqrt(sample_size)
    # Print for debugging
    #print(f"Standard Deviation: {std_dev}, Sample Size: {sample_size}, SEM: {sem}")


    # Calculate the confidence interval
    confidence_level = 0.95
    margin_of_error = sem * stats.norm.ppf((1 + confidence_level) / 2)

    return mean, margin_of_error

def update(results, *values):
    """Update result from training and validation for each epoch.
    Append provided values to results dictionary.

    Returns:
        A dictionary of loss and passing rate metrics.
    """
    for (key, _), value in zip(results.items(), values):
        results[key].append(value)

    return results

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
    


def plot_data(batch_data, batch_fake, batch_water, batch_density, data_name, save_folder_path, epoch=1):
    n = 32  # Number of samples to plot
    voxel = np.arange(256) 
    #Length in mm = Original_Voxel_Number × Voxel Size
    #x_vals = (voxel*voxel_size -181) #- (-110) Use depth[mm] from source as x-axis
    # 'Data4':
    voxel_size = 0.666016*2
    x_vals = (voxel*voxel_size -170.5) #- (-110) #Use depth[mm] from source as x-axis
    #'Data1':
    #voxel_size = 0.707031*2
    #x_vals = (voxel*voxel_size -181)
    
 
    # Define the mm values for the ticks within the range of x_vals
    min_x, max_x = min(x_vals), max(x_vals)
    tick_mm_values = np.arange(-150, 150, 25)  # Adjust the start and end according to your needs
    tick_mm_values = tick_mm_values[(tick_mm_values >= min_x) & (tick_mm_values <= max_x)]

    average_delta_list =[]
    for i in range(min(n,len(batch_data))):
        fig = plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 

        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1], sharex=ax0)

        real_dose = batch_data[i].squeeze().cpu().numpy()
        fake_dose = batch_fake[i].squeeze().cpu().numpy()
        water_dose = batch_water[i].squeeze().cpu().numpy()
        density = batch_density[i].squeeze().cpu().numpy()
        max_index = np.unravel_index(real_dose.argmax(), real_dose.shape)

        # Extract slices
        #real_dose_slice = np.sum(real_dose[7:9, 7:9, :], axis=(0, 1)) 
        real_dose_slice = real_dose[int(max_index[0]), int(max_index[1]), :] 
        #real_dose_slice =real_dose_slice
        #real_dose_slice = np.sum(real_dose, axis=(0, 1)) 
        #fake_dose_slice = np.sum(fake_dose[7:9, 7:9, :], axis=(0, 1)) 
        fake_dose_slice = fake_dose[int(max_index[0]), int(max_index[1]), :] 
        #fake_dose_slice = fake_dose_slice
        #fake_dose_slice = np.sum(fake_dose, axis=(0, 1)) 
        water_dose_slice = water_dose[8, 8, :]
        density_slice = density[int(max_index[0]), int(max_index[1]), :]
        

        ax0.plot(x_vals, real_dose_slice, label='Simulated dose')
        ax0.plot(x_vals, fake_dose_slice, label='Generated dose', linestyle='--')
        #ax0.plot(x_vals, water_dose_slice, label='Dose in water')
        ax0.set_ylabel('Dose/Dose_max')
        #ax0.set_ylabel('Dose [Gy]')
        ax0.legend(loc='upper left')
        ax0.set_title(f'{data_name[i]}')

        # Create a twin axis for the density
        ax2 = ax0.twinx()
        ax2.plot(x_vals, density_slice, label='Density', color='gray', linestyle='--')
        ax2.set_ylabel('Normalized Density')
        ax2.legend(loc='upper right')

        # Plotting the delta percent on the bottom subplot
        max_real_dose = np.max(real_dose_slice)
        #delta_percent = 100 * (fake_dose_slice - real_dose_slice) / max_real_dose
        delta_percent = 100 * (fake_dose_slice - real_dose_slice) * real_dose_slice
        # Calculate the average
        average_delta_percent = np.mean(abs(delta_percent))
        average_delta_list.append(average_delta_percent)
        ax1.scatter(x_vals, delta_percent, label='Delta %', s=2)
        # Set the custom tick labels
        ax1.set_xticks(tick_mm_values)
        ax1.set_xticklabels([f"{mm:.0f}" for mm in tick_mm_values])

        ax1.set_ylabel('ΔDose [%]')
        ax1.set_xlabel('Depth [mm]')
        for delta in range(-40, 40, 10):  # Creates lines at -100%, -75%, ..., 100%
            ax1.axhline(y=delta, color='gray', linestyle='--', linewidth=0.5)

        ax1.set_ylim(-40, 40)  # Adjust the y-axis limits as needed

        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.subplots_adjust(hspace=.0)
        plt.suptitle('Comparison of simulated and generated dose along the beam direction')

        # Save each sample's plot as a separate file
        individual_save_filename = f'epoch{epoch}_{data_name[i][:-4]}.png'
        plt.savefig(os.path.join(save_folder_path, individual_save_filename))
        plt.close(fig)

    print("average : ",sum(average_delta_list)/len(average_delta_list))
    

# Use this function with your data like this:
# plot_data(your_batch_data, your_batch_fake, your_data_names, your_save_folder_path)


def plot_data_xyz(batch_data, batch_fake, data_name, save_folder_path):
    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    save_filename = f'plot_{timestamp}.png'
    n = 10  # Number of samples to plot
    VOXELNUMBER_X, VOXELNUMBER_Y, VOXELNUMBER_Z = 128, 16, 16

    x_vals = np.arange(VOXELNUMBER_X)
    y_vals = np.arange(VOXELNUMBER_Y)
    z_vals = np.arange(VOXELNUMBER_Z)
    index_list = []

    fig, axs = plt.subplots(n, 4, figsize=(20, 5 * n))  # Adjust the subplots

    for i in range(n):
        real_dose = batch_data[i].squeeze().cpu().numpy()
        fake_dose = batch_fake[i].squeeze().cpu().numpy()
        max_index = np.unravel_index(real_dose.argmax(), real_dose.shape)
        index_list.append(max_index)

        # Extract slices
        real_dose_slice = real_dose[int(max_index[0]), int(max_index[1]), :]
        fake_dose_slice = fake_dose[int(max_index[0]), int(max_index[1]), :]

        # Real data plots
        axs[i, 0].plot(z_vals, real_dose[:, int(max_index[1]), int(max_index[2])], label='Real')
        axs[i, 1].plot(y_vals, real_dose[int(max_index[0]), :, int(max_index[2])], label='Real')
        axs[i, 2].plot(x_vals, real_dose_slice, label='Real')

        # Fake data plots
        axs[i, 0].plot(z_vals, fake_dose[:, int(max_index[1]), int(max_index[2])], label='Fake', linestyle='--')
        axs[i, 1].plot(y_vals, fake_dose[int(max_index[0]), :, int(max_index[2])], label='Fake', linestyle='--')
        axs[i, 2].plot(x_vals, fake_dose_slice, label='Fake', linestyle='--')


        # Calculate delta_percent
        delta_percent = np.where(real_dose_slice != 0, 
                                100 * (fake_dose_slice - real_dose_slice) / real_dose_slice, 
                                np.nan)

        axs[i, 3].plot(x_vals, delta_percent, label='Delta%')

        for j in range(4):
            axs[i, j].set_yscale('linear')
            axs[i, j].set_ylabel('Dose [Gy]' if j != 3 else 'Delta %')
            axs[i, j].set_ylim(-100, 100) if j == 3 else axs[i, j].set_ylim(0.0, 1.0)
            axs[i, j].legend()

        axs[i, 0].set_xlabel('Z [mm]')
        axs[i, 1].set_xlabel('Y [mm]')
        axs[i, 2].set_xlabel('X [mm] (Beam direction)')
        axs[i, 3].set_xlabel('X [mm] (Beam direction)')
        axs[i, 0].set_title(f'Sample {i+1}: {data_name[i]}')
        axs[i, 3].set_title(f'Delta% for Voxel: {max_index}')

    plt.suptitle('Comparison of Real and Generated Data Doses')
    plt.savefig(os.path.join(save_folder_path, save_filename))
    plt.close(fig)

def plot_slice(batch_data, batch_fake, batch_density, batch_data_name, save_folder_path, epoch=1):
    '''Comparison of Simulated and Generated Dose Distribution'''
    n = 8  # Number of samples to plot
    voxel = np.arange(256) 
    #Length in mm = Original_Voxel_Number × Voxel Size
    voxel_size = 0.666016*2
    x_vals = (voxel*voxel_size -170.5) #- (-110) #Use depth[mm] from source as x-axis
    voxel_size_z = 1.25*2
 
    # Define the mm values for the ticks within the range of x_vals
    min_x, max_x = min(x_vals), max(x_vals)
    tick_mm_values = np.arange(-150, 150, 20)  # Adjust the start and end according to your needs
    tick_mm_values = tick_mm_values[(tick_mm_values >= min_x) & (tick_mm_values <= max_x)]

    # Determine global min and max values for the entire batch_data
    data_max = np.max([sample.max() for sample in batch_data])
    density_max = np.max([sample.max() for sample in batch_density])
    # Common settings for colormap
    density_colormap_setting = {'cmap': 'gray', 'alpha': 0.7, 'vmin': 0, 'vmax': density_max}
    colormap_setting = {'cmap': 'inferno', 'alpha': 0.7, 'vmin': 0, 'vmax': data_max}
    delta_colormap_setting = {'cmap': 'YlOrRd', 'alpha': 0.7, 'vmin': 0, 'vmax': 15}
    # Create the 'slice' directory
    slice_dir = save_folder_path / 'slice'
    slice_dir.mkdir(parents=True, exist_ok=True)


    for i in range(min(n,len(batch_data))):
        data_sample_real = batch_data[i].squeeze().cpu().numpy()
        data_sample_fake = batch_fake[i].squeeze().cpu().numpy()
        density_sample = batch_density[i].squeeze().cpu().numpy()
        data_name = batch_data_name[i]

        max_index = np.unravel_index(data_sample_real.argmax(), data_sample_real.shape)

        # Extract slices
        #real_dose_slice =np.sum(data_sample_real, axis=(0, 1)) 
        #fake_dose_slice =np.sum(data_sample_fake, axis=(0, 1)) 
        max_real_dose = np.max(data_sample_real)
        delta = 100 * abs(data_sample_fake - data_sample_real) / max_real_dose
        

        y = extract_from_filename(data_name, 'y')
        z = extract_from_filename(data_name, 'z') 
        # Calculate the extent for each axis
        x_extent = [-170.5, 170.5]
        y_extent = [y - voxel_size * density_sample.shape[1] / 2, y + voxel_size * density_sample.shape[1] / 2]
        z_extent = [z - voxel_size_z * density_sample.shape[0] / 2, z + voxel_size_z * density_sample.shape[0] / 2]

        max_index = np.unravel_index(data_sample_real.argmax(), data_sample_real.shape)

        fig = plt.figure(figsize=(20, 6))
        gs = GridSpec(3, 3, width_ratios=[1, 5, 5], height_ratios=[1, 1, 1], figure=fig)  # Adjust the width_ratios as needed

        # Create subplots using GridSpec
        axs = []
        for r in range(3):
            for c in range(3):
                axs.append(fig.add_subplot(gs[r, c]))

        # Plot real data slices
        axs[0].imshow(density_sample[:, :, max_index[2]], aspect='auto', **density_colormap_setting, extent=y_extent + z_extent)
        axs[0].imshow(data_sample_real[:, :, max_index[2]], aspect='auto', **colormap_setting, extent=y_extent + z_extent)

        axs[1].imshow(density_sample[:, max_index[1], :], **density_colormap_setting, extent=x_extent + z_extent)
        axs[1].imshow(data_sample_real[:, max_index[1], :], **colormap_setting, extent=x_extent + z_extent)
        axs[1].set_ylabel('Z [mm]')

        axs[2].imshow(density_sample[max_index[0], :, :], **density_colormap_setting, extent=x_extent + y_extent)
        img_data = axs[2].imshow(data_sample_real[max_index[0], :, :], **colormap_setting, extent=x_extent + y_extent)
        axs[2].set_ylabel('Y [mm]')

        # Plot fake data slices
        axs[3].imshow(density_sample[:, :, max_index[2]], aspect='auto', **density_colormap_setting, extent=y_extent + z_extent)
        axs[3].imshow(data_sample_fake[:, :, max_index[2]], aspect='auto', **colormap_setting, extent=y_extent + z_extent)

        axs[4].imshow(density_sample[:, max_index[1], :], **density_colormap_setting, extent=x_extent + z_extent)
        axs[4].imshow(data_sample_fake[:, max_index[1], :], **colormap_setting, extent=x_extent + z_extent)
        axs[4].set_ylabel('Z [mm]')

        axs[5].imshow(density_sample[max_index[0], :, :], **density_colormap_setting, extent=x_extent + y_extent)
        img_data_fake = axs[5].imshow(data_sample_fake[max_index[0], :, :], **colormap_setting, extent=x_extent + y_extent)
        axs[5].set_ylabel('Y [mm]')

        axs[6].imshow(density_sample[:, :, max_index[2]], aspect='auto', **density_colormap_setting, extent=y_extent + z_extent)
        axs[6].imshow(delta[:, :, max_index[2]], aspect='auto', **delta_colormap_setting, extent=y_extent + z_extent)
       

        axs[7].imshow(density_sample[:, max_index[1], :], **density_colormap_setting, extent=x_extent + z_extent)
        delta_img_data = axs[7].imshow(delta[:, max_index[1], :], **delta_colormap_setting, extent=x_extent + z_extent)
        axs[7].set_ylabel('Z [mm]')

        axs[8].imshow(density_sample[max_index[0], :, :], **density_colormap_setting, extent=x_extent + y_extent)
        axs[8].imshow(delta[max_index[0], :, :], **delta_colormap_setting, extent=x_extent + y_extent)
        axs[8].set_ylabel('Y [mm]')

          # For setting x-ticks and x-tick labels for each subplot
        for ax_index in range(1, 9):  # axs[1] to axs[5] correspond to the middle and right columns
            axs[ax_index].set_xticks(tick_mm_values)
            axs[ax_index].set_xticklabels([f"{mm:.0f}" for mm in tick_mm_values])
            axs[ax_index].set_xlabel('X [mm]')
            if ax_index % 3 != 0:  # Only set x-limits for the middle and right column plots
                axs[ax_index].set_xlim(-170, 170)

        tick_mm_values_y = np.arange(y-7*voxel_size, y+7*voxel_size, 5)
        tick_mm_values_z = np.arange(z-7*voxel_size_z, z+7*voxel_size_z, 10)
        for ax_index in [0, 3, 6]:
            axs[ax_index].set_xticks(tick_mm_values_y)
            axs[ax_index].set_xticklabels([f"{mm:.0f}" for mm in tick_mm_values_y])
            #axs[ax_index].set_yticks(tick_mm_values_z)
            #axs[ax_index].set_yticklabels([f"{mm:.0f}" for mm in tick_mm_values_z])
            axs[ax_index].set_xlabel('Y [mm]')
            axs[ax_index].set_ylabel('Z [mm]')
            axs[ax_index].set_xlim(y-7*voxel_size, y+7*voxel_size)
            #axs[ax_index].set_ylim(z-7*voxel_size_z, z+7*voxel_size_z)
        for ax_index in [1, 4, 7]:
            axs[ax_index].set_yticks(tick_mm_values_z)
            axs[ax_index].set_yticklabels([f"{mm:.0f}" for mm in tick_mm_values_z])
        for ax_index in [2, 5, 8]:
            axs[ax_index].set_yticks(tick_mm_values_y)
            axs[ax_index].set_yticklabels([f"{mm:.0f}" for mm in tick_mm_values_y])


        axs[1].set_title('Simulated Dose')
        axs[4].set_title('Generated Dose')
        axs[7].set_title('Delta')
        '''

        # Set x-ticks and x-tick labels for each subplot
        for row in range(2):  # Two rows in subplot grid
            for col in range(1,3):  # Three columns in subplot grid
                axs[row, col].set_xticks(tick_mm_values)
                axs[row, col].set_xticklabels([f"{mm:.0f}" for mm in tick_mm_values])
                axs[row, col].set_xlabel('X [mm]')
                axs[row, col].set_xlim(-170, 170)

        tick_mm_values_y = np.arange(y-7*voxel_size, y+7*voxel_size,2.5)
        tick_mm_values_z = np.arange(z-7*voxel_size_z, z+7*voxel_size_z,2.5)
        for row in range(2):  # Two rows in subplot grid
            for col in range(1):  # Three columns in subplot grid
                axs[row, col].set_xticks(tick_mm_values_y)
                axs[row, col].set_xticklabels([f"{mm:.0f}" for mm in tick_mm_values_y])
                axs[row, col].set_yticks(tick_mm_values_z)
                axs[row, col].set_yticklabels([f"{mm:.0f}" for mm in tick_mm_values_z])
                axs[row, col].set_title('Generated Dose')
                axs[row, col].set_xlabel('Y [mm]')
                axs[row, col].set_ylabel('Z [mm]')
                axs[row, col].set_xlim(y-7*voxel_size, y+7*voxel_size)
                axs[row, col].set_ylim(z-4*voxel_size_z, z+4*voxel_size_z)

        '''
        # Add colorbar
        #fig.colorbar(img_data, ax=[axs[j] for j in range(3)], fraction=0.046, pad=0.04).set_label('dose / dose_max')
        #fig.colorbar(img_data_fake, ax=[axs[j] for j in range(3, 6)], fraction=0.046, pad=0.04).set_label('dose / dose_max')
        # Add colorbar for the real data plots, make sure to place it at the far right
        cbar_ax = fig.add_axes([0.85, 0.65, 0.12, 0.02])  # [left, bottom, width, height]
        fig.colorbar(img_data, cax=cbar_ax, orientation='horizontal').set_label('dose / dose_max')
        cbar_ax = fig.add_axes([0.85, 0.35, 0.12, 0.02])  # [left, bottom, width, height]
        fig.colorbar(delta_img_data, cax=cbar_ax, orientation='horizontal').set_label('delta (%)')

        #plt.suptitle('Comparison of Simulated and Generated Dose Distribution')
        plt.suptitle(f'{data_name}')
        # Adjust subplots to have the same height
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.tight_layout() 
        # Save the plot
        individual_save_filename = f'epoch{epoch}_{data_name[:-4]}.png'
        plt.savefig(slice_dir / individual_save_filename)
        plt.close(fig)


# Example usage:
# plot_slice(batch_data, batch_fake, batch_density, data_name, 'path/to/save_folder')



