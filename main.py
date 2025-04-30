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
######################################################################
torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device :',device)

torch.backends.cudnn.enabled=True # comment BatchNorm1d -> CUDNN_STATUS_NOT_SUPPORTED
# Enable cuDNN benchmark mode
torch.backends.cudnn.benchmark = True


# Define thresholds and parameters
TREND_CHECK_WINDOW = 10  # Check the trend over the last 5 epochs
MSE_DECREASE_THRESHOLD = 0.01  # Allowable MSE decrease threshold
DELTA_DECREASE_THRESHOLD = 0.01  # Allowable Delta decrease threshold



def is_metric_decreasing(metric_history, window_size, threshold):
    if len(metric_history) < window_size:
        return False
    recent_metrics = metric_history[-window_size:]
    return all(recent_metrics[i] <= recent_metrics[i - 1] + threshold for i in range(1, len(recent_metrics)))

def convergence_criteria_met(mse, delta):
    # Ensure there's enough data to check for a trend
    if len(mse) < TREND_CHECK_WINDOW:
        return False

    # Check if MSE and Delta are decreasing for both generator and critic
    mse_decreasing = is_metric_decreasing(mse, TREND_CHECK_WINDOW, MSE_DECREASE_THRESHOLD)
    delta_decreasing = is_metric_decreasing(delta, TREND_CHECK_WINDOW, DELTA_DECREASE_THRESHOLD)

    # Check if all metrics are showing a decreasing trend
    return mse_decreasing and delta_decreasing 

def should_terminate_early(epoch, gen_loss_history, critic_loss_history, patience=15, min_delta=0.01):
    """
    Check if the training should be terminated early based on loss history.

    Args:
    - epoch (int): Current epoch number.
    - gen_loss_history (list of float): History of generator loss values.
    - critic_loss_history (list of float): History of critic loss values.
    - patience (int): Number of epochs to wait for improvement before terminating.
    - min_delta (float): Minimum change in loss to qualify as an improvement.

    Returns:
    - bool: True if training should be terminated early, False otherwise.
    """

    # Check if we have enough data to consider early stopping
    if epoch < patience:
        return False

    # Check the last 'patience' epochs for both generator and critic
    recent_gen_losses = gen_loss_history[-patience:]
    recent_critic_losses = critic_loss_history[-patience:]

    # If loss has not decreased by at least min_delta, terminate early
    if (min(recent_gen_losses) > recent_gen_losses[0] - min_delta and
        min(recent_critic_losses) > recent_critic_losses[0] - min_delta):
        return True

    return False



class EarlyStopping:
    def __init__(self, model, log_dir, base_filename='model', patience=25, min_delta=0):
        """
        Early stops the training if MSE doesn't improve after a given patience.
        Additionally saves the model at its best state with MSE in the filename.
        
        :param model: The model to be saved when MSE improves.
        :param log_dir: Directory where the model will be saved.
        :param base_filename: Base filename for saving the model. MSE will be appended to this.
        :param patience: (int) How many epochs to wait after last time MSE improved.
                         Default: 25
        :param min_delta: (float) Minimum change in the monitored quantity 
                          to qualify as an improvement. Default: 0
        """
        self.model = model
        self.log_dir = log_dir
        self.base_filename = base_filename
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, mse):
        score = -mse

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.min_delta:
            self.counter += 1
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def load_data_from_folder(saved_dataset):
    with open(saved_dataset, 'rb') as file:
        return pickle.load(file)
    

# Define high positive threshold for generator loss
GEN_LOSS_THRESHOLD = 200.0  # Example value, adjust based on observations

# Define low negative threshold for critic loss
CRITIC_LOSS_THRESHOLD = -30.0  # Example value, adjust based on observations

#######################################################################
########################## DATA LOADING ###############################

BATCH_SIZE = 32 
dataset_dir = "/home/tappay01/new_data/16011153/"

train_dataset = dataset_dir + 'train_dataset_aug.pkl'
validation_dataset =  dataset_dir + 'val_dataset.pkl'
test_dataset = dataset_dir + 'test_dataset.pkl'
with open(train_dataset, 'rb') as file:
   train_subset = pickle.load(file)
with open(validation_dataset, 'rb') as file:
   val_subset = pickle.load(file)
with open(test_dataset, 'rb') as file:
   test_subset = pickle.load(file)


# Turn train, val and test custom Dataset into DataLoader's
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8) #creates 8 worker processes to load the data in parallel. 
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)


print("Total number of samples in the train dataset:", len(train_subset))
print("Number of batches:", len(train_loader))
print("Total number of samples in the validation dataset:", len(val_subset))
print("Number of batches:", len(val_loader))
print("Total number of samples in the test dataset:", len(test_subset))
print("Number of batches:", len(test_loader))

# Define the range of learning rates
learning_rates_gen = [0.003]#0.003
learning_rates_critic = [0.00002] #[0.00001]
# Define the range of alpha of gradient penalty
LAMBDA = 10
CRITIC_ITERATION= 5                    

                      

for lr_gen in learning_rates_gen:
    for lr_critic in learning_rates_critic:
        experiments = [
                #{"optimizer": "Adam", "lr_gen": lr_gen, "lr_critic": lr_critic},
                {"optimizer": "RMSprop", "lr_gen": lr_gen, "lr_critic": lr_critic},
                        ]
        for exp_params in experiments:    
            ######################### CREATE LOG DIRECTORY ########################
            # Create target directory
            dir_name = f"LRG_{lr_gen}_LRC_{lr_critic}_L_{LAMBDA}_opt_{exp_params['optimizer']}"
            timestamp = datetime.now().strftime("%d%m%H%M")
            log_dir = f"/home/tappay01/test/runs/{timestamp}_{dir_name}/"
            print(log_dir)
            target_dir = Path(log_dir)
            target_dir.mkdir(parents=True,exist_ok=True)

            val_dir = target_dir / 'validation'
            val_dir.mkdir(parents=True, exist_ok=True)

            test_dir = target_dir/ 'test'
            test_dir.mkdir(parents=True, exist_ok=True)
            
            ######################### HYPERPARAMETER ##############################
            Z_DIM =100
            NUM_EPOCHS = 1000
            #ALPHA         
            LAMBDA = 10 #Lambda for gradient penalty 
            
            print('Learning rate Gen :',lr_gen)
            print('Learning rate Critic :',lr_critic)            
            print('LAMBDA_GP for gradient penalty :',LAMBDA)
            print("BATCH_SIZE :",BATCH_SIZE)
            output_file_path = target_dir/"parameter.txt"
            print('CRITIC_ITERATIONS :',CRITIC_ITERATION)
            with open(output_file_path, "w") as file:
                file.write("LEARNING_RATE_G: {}\n".format(lr_gen))
                file.write("LEARNING_RATE_C: {}\n".format(lr_critic))
                file.write("LAMBDA: {}\n".format(LAMBDA))
                file.write("BATCH_SIZE : {}\n".format(BATCH_SIZE))
                file.write("CRITIC_ITERATIONS : {}\n".format(CRITIC_ITERATION))
            #######################################################################
            ########################### INITIALIZE MODELS ##########################
            gen = Generator()
            initialize_weights(gen)
            gen = gen.to(device)
            critic = Critic3d()
            initialize_weights(critic)
            critic = critic.to(device)
            for param in gen.parameters():
                param= param.to(device)
            for param in critic.parameters():
                param= param.to(device)
            '''
            # Load the model
            #if device == 'cpu':
            model_g_path = "/home/tappay01/test/runs/01011438_LRG_0.009_LRC_2e-05_L_10_CI_5/generator_epoch441.pth"
            model_c_path = "/home/tappay01/test/runs/01011438_LRG_0.009_LRC_2e-05_L_10_CI_5/critic_epoch441.pth"
            if torch.cuda.is_available() == False:
                state_dict = torch.load(model_g_path, map_location=torch.device('cpu')) # to edit when use gpu
                gen.load_state_dict(state_dict)
                critic_state_dict = torch.load(model_c_path, map_location=torch.device('cpu')) # to edit when use gpu
                critic.load_state_dict(critic_state_dict)
            else:
                gen = gen.to(device)
                critic = critic.to(device)
                state_dict = torch.load(model_g_path)
                gen.load_state_dict(state_dict)
                critic_state_dict = torch.load(model_c_path)
                critic.load_state_dict(critic_state_dict)
            '''

            #######################################################################
            ################################ TRAIN MODELS ##########################
            #Set up optimizers for generator and critic
            # Choose the optimizer based on the experiment
            if exp_params["optimizer"] == "Adam":
                opt_gen = optim.Adam(gen.parameters(), lr=exp_params["lr_gen"], betas=(0.0, 0.9))
                opt_critic = optim.Adam(critic.parameters(), lr=exp_params["lr_critic"], betas=(0.0, 0.9))
            elif exp_params["optimizer"] == "RMSprop":
                opt_gen = optim.RMSprop(gen.parameters(), lr=exp_params["lr_gen"])
                opt_critic = optim.RMSprop(critic.parameters(), lr=exp_params["lr_critic"])
            else:
                raise ValueError(f"Unsupported optimizer: {exp_params['optimizer']}")


            # Variable to track convergence
            has_converged = False

            early_stopping = EarlyStopping(gen, log_dir, base_filename='gen_model', patience=50, min_delta=0.005)

            # Create empty results dictionary
            results_keys = [
                "epoch_loss_gen", "epoch_loss_critic",
                "mse", "val_mae"]

            results = {key: [] for key in results_keys}

            # Initialize variables to keep track of the best model and its performance
            best_gen_state_dict = None
            best_critic_state_dict = None
            best_performance = 0.0
            best_val_passing_rate3 = -float('inf')
            best_mse = float('inf')
            model_g_name = ''
            model_c_name =''

            # loop over epochs
            #print("[INFO] training the network...")
            startTime = time.time()
            
            for e in tqdm(range(NUM_EPOCHS)): 
                reason = ""
                #report_gpu()
                #epoch_loss_gen,epoch_loss_critic,train_passing_rate1, train_error1 ,train_passing_rate3, train_error3 = train_step(gen, critic,train_loader,opt_gen,opt_critic,LAMBDA_GP, device,CRITIC_ITERATION,ALPHA) 
                epoch_loss_gen,epoch_loss_critic = train_step(gen, critic,train_loader,opt_gen,opt_critic,LAMBDA, device,CRITIC_ITERATION) 
                                                                                                              


                #val_passing_rate1, val_error1, val_passing_rate3, val_error3 ,mse, mse_error = val_step(gen, critic, val_loader, device, val_dir, e)
                mse, val_mae  = val_step(gen, critic, val_loader, device, val_dir, e)
                

                # Update results dictionary with this epoch's values
                results = update(results, 
                                    epoch_loss_gen, epoch_loss_critic, 
                                   mse, val_mae)
                
                
                # print the model training and validation information
                #print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
                #print("Train loss generator: {:.4f}, loss critic: {:.4f}".format(epoch_loss_gen, epoch_loss_critic ))
                #print("train delta : {:.6f}".format(train_delta))

                # Early termination if not converging
                # Call early stopping logic
                '''
                early_stopping(mse)
                if early_stopping.early_stop:
                    reason = f"Stopping early due to no improvement in MSE. at epoch {e}"
                    break
                '''
                # Check for divergence
                if e>=10:
                    if epoch_loss_gen > GEN_LOSS_THRESHOLD or epoch_loss_critic < CRITIC_LOSS_THRESHOLD:
                        reason = f"Training diverged at epoch {e}: generator loss = {epoch_loss_gen}, critic loss = {epoch_loss_critic}"
                        break 

                # During your training loop or validation check
                if mse < best_mse:
                    #print(f"[INFO] Found new lower MSE: {mse:.6f}")
                    model_g_name = f"MSE_{mse}_generator_epoch{e}.pth"
                    model_c_name = f"MSE_{mse}_critic_epoch{e}.pth"

                    # Update the best model state dicts
                    best_gen_state_dict = gen.state_dict()
                    best_critic_state_dict = critic.state_dict()

                    # Update the best MSE
                    best_mse = mse

            
            # save the model at last epoch
            model_g_path = os.path.join(target_dir, f"generator_epoch{e}.pth")
            model_c_path = os.path.join(target_dir, f"critic_epoch{e}.pth")
            torch.save(obj=gen.state_dict(), f=model_g_path)
            torch.save(obj=critic.state_dict(), f=model_c_path)

            if model_g_name:
            # After the loop, save the best state_dicts
                model_g_path = os.path.join(target_dir, model_g_name)
                model_c_path = os.path.join(target_dir, model_c_name)
                torch.save(obj=best_gen_state_dict, f=model_g_path)
                torch.save(obj=best_critic_state_dict, f=model_c_path)


            # display the total time needed to perform the training
            endTime = time.time()
            #print("[INFO] total time taken to train the model: {:.2f}s".format(
            #    endTime - startTime))
            

            #######################################################################
            ####################### VISUALIZE TRAINING RESULTS ###################

            # Plot loss per epoch
            plt.figure(figsize=(10,5))
            plt.title("Generator and Critic Loss During Training Per Epoch ")
            plt.plot(results["epoch_loss_gen"],label="Generator_loss")
            plt.plot(results["epoch_loss_critic"],label="Critic")
            plt.xlabel("number of epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(log_dir + "loss.png")
            plt.clf()

            # Plot passing MSE per epoch
            plt.figure(figsize=(10,5))
            plt.title("Mean Squared Error (MSE)")
            plt.plot(results["mse"], label="avg_MSE",  color='red', marker='.', markersize=1)
            plt.xlabel("number of epoch")
            plt.ylabel("Error")
            plt.ylim(0, 10)
            plt.legend()
            plt.savefig(log_dir + "MSE.png")
            plt.clf()

            save(results, log_dir + 'results.pkl')

            ############################PERFORMANCE TEST############################
            
            #test_model(model_g_path, device, test_subset, test_dir)
