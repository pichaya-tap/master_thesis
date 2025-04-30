"""
Contains functions for training and validating a model.
"""
import torch
from model import gradient_penalty
from utils import plot_data, plot_slice, cal_passing_rate, cal_epoch_rate, mean_absolute_error
import gc
import torch.nn as nn
import torch.nn.functional as F

def report_gpu():
   print(torch.cuda.list_gpu_processes())
   gc.collect()
   torch.cuda.empty_cache()
   print(f"{torch.cuda.memory_allocated()/(1024)} Kb")

import torch




def train_step(gen,
               critic,
               train_loader: torch.utils.data.DataLoader, 
               opt_gen: torch.optim.Optimizer,
               opt_critic: torch.optim.Optimizer,
               LAMBDA_GP,
               device: torch.device,
               CRITIC_ITERATIONS=5,
               ):         
                   
    """Trains a PyTorch model for a single epoch.
    Turns a model to training mode and then
    runs through all of the required training steps.

    Args:
        gen: A PyTorch generator model to be trained.
        critic: A PyTorch critic model to be trained.
        train_loader: A DataLoader instance for the model to be trained on.
        opt_gen: An optimizer to help minimize the loss function of generator.
        opt_critic: An optimizer to help minimize the loss function of discriminator.
        LAMBDA_GP : Lambda for gradient penalty
        device: A target device to compute on (e.g. "cuda" or "cpu")

    Returns:
        Training loss per epoch and passing rate 1%       
        (epoch_loss_gen,  epoch_loss_critic, epoch_passing_rate_1).
    """
    #report_gpu()
    # Put model in train mode
    gen.train()
    critic.train()
    # Initialize train performance metrics values
    generator_losses  = []
    critic_losses = []
    passing_rates1 = []
    passing_rates3 = []
    total_delta = 0
    total_samples = 0

     # Update critics = CRITIC_ITERATIONS times before update the generator

    # Loop through data loader data batches
    for batch_idx, (real, cond, water_tensor, density_tensor, data_name) in enumerate(train_loader): 
        #print(f"Processing train batch {batch_idx}")
        cur_batch_size = real.shape[0]    
        # Send data to target device
        # Use .float() because of RuntimeError: expected scalar type Double but found Float
        real = real.to(device)  
        cond = cond.to(device)        

                     
        ########## Train Critic: max E[critic(real)] - E[critic(fake)]###########
        ########## equivalent to minimizing the negative of that #################
        # Update critics = CRITIC_ITERATIONS times before update the generator
        mean_iteration_critic_loss = 0
        for _ in range(CRITIC_ITERATIONS): 
            #report_gpu()
            # print('Training Critic')
            fake = gen(cond)
                           
            critic_fake = critic(fake.detach(), cond).reshape(-1)  
            #print("critic_fake ",critic_fake)
            critic_real = critic(real, cond).reshape(-1)   
            #print("critic_real ",critic_real)  
            gp = gradient_penalty(critic, real, fake.detach(), cond, device=device)                
           
            # Calculate  and accumulate loss
            loss_critic = -(torch.mean(critic_real)- torch.mean(critic_fake)) + LAMBDA_GP*gp
            
            #print("loss_critic :",-float(torch.mean(critic_real)- torch.mean(critic_fake)) ,"gradient penalty :",float(LAMBDA_GP*gp))
            # Keep track of the average critic loss in this batch
            mean_iteration_critic_loss += loss_critic.item() / CRITIC_ITERATIONS
            # Optimizer zero grad to zero out any previously accumulated gradients
            critic.zero_grad()
            # Perform backpropagation
            loss_critic.backward(retain_graph=True) # True, able to reuse
            # Optimizer step to update model parameters
            opt_critic.step()       
            del(fake)
        critic_losses += [mean_iteration_critic_loss]
           

        ########## Train Generator: min -E[critic(gen_fake)] ##########
        ###############################################################
        #print('Training Generator')
        # Optimizer zero grad
        gen.zero_grad()
        fake_2 = gen(cond).float().to(device)
        crit_fake_pred = critic(fake_2, cond).reshape(-1)
        del cond


        # Calculate dose ratio
        dose_ratio = real

        #print("crit_fake_pred ",crit_fake_pred)
        loss_gen = -1.*torch.mean(crit_fake_pred) 

        #loss_gen = torch.mean(critic_real)- torch.mean(critic_fake)   
        # Loss backward
        loss_gen.backward()
        # Optimizer step
        opt_gen.step()
        # Keep track of the average generator loss
        generator_losses  += [loss_gen.item()]

        #################### Performance metric ######################
        #batch_passing_rates1 = cal_passing_rate(0.01, real.detach(), fake_2.detach())
        #batch_passing_rates3 = cal_passing_rate(0.03, real.detach(), fake_2.detach())
       # Keep track of the passing_rate
        #passing_rates1 += batch_passing_rates1       
        #passing_rates3 += batch_passing_rates3  

        total_samples += real.size(0)

        del fake_2
        del real
    # Calculate loss per epoch and passing rate 
    epoch_loss_gen = sum(generator_losses)/len(generator_losses)
    epoch_loss_critic = sum(critic_losses)/len(critic_losses)
    # Calculate epoch statistics
    #mean_passing_rate1, margin_of_error1 = cal_epoch_rate(passing_rates1)
    #mean_passing_rate3, margin_of_error3 = cal_epoch_rate(passing_rates3)


    return epoch_loss_gen,  epoch_loss_critic
###############################End of: def train_step ####################################################
    
def val_step(gen,
               critic,
               val_loader: torch.utils.data.DataLoader, 
               device: torch.device,
                log_dir,
                current_epoch):                

    
    """Test the model for a single epoch.
    Turns a target model to "eval" mode and then performs a forward pass on a validation dataset.

    Args:
        gen: A generator model to be validated.
        critic: A critic model to be validated.
        val_loader: A DataLoader instance for the model to be validated on.
        device: A target device to compute on (e.g. "cuda" or "cpu")

    Returns:
    Average validation passing rate %.
    """
    # Put model in eval mode
    gen.eval() 
    critic.eval()
     
    # Setup train loss values
    passing_rates1 = []
    passing_rates3 = []
    total_maes = []


    total_mse_loss = 0

    total_samples = 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop over the validation set
        for batch_idx, (real,cond, water_tensor, density_tensor, data_name) in enumerate(val_loader): 
            cur_batch_size = real.shape[0]    
            # print(f"Processing val batch {batch_idx}")
            # send the input to the device
            real = real.to(device) 
            cond = cond.to(device) 
 
            # Forward pass
            fake = gen(cond).float().to(device)

            batch_passing_rates1 = cal_passing_rate(0.01, real.detach(), fake.detach())
            batch_passing_rates3 = cal_passing_rate(0.03, real.detach(), fake.detach())
            # Keep track of the passing_rate
            passing_rates1 += batch_passing_rates1
            passing_rates3 += batch_passing_rates3
            # Print passing rate occasionally
            #print(f"Batch {batch_idx}/{len(val_loader)}") 
            #print(f"Val passing rate 3% :{batch_passing_rates}") 
            batch_maes = mean_absolute_error(real.detach(), fake.detach())
            total_maes += batch_maes
            total_samples += real.size(0)

            ################### Visualization ######################
            if current_epoch %100 == 0 and batch_idx == 0: # To change to some number
                with torch.no_grad():
                    plot_data(real.detach().cpu(), fake.detach().cpu(), water_tensor.detach().cpu(),density_tensor.detach().cpu(), data_name, log_dir, current_epoch )
                    #plot_slice(real.detach().cpu(), fake.detach().cpu(), density_tensor.detach().cpu() ,data_name, log_dir, current_epoch)
            
            #################### Performance metric ######################
            # Compute the MSE loss
            mse_loss = F.mse_loss(fake, real,  reduction='sum')
            # Accumulate the loss
            total_mse_loss += mse_loss.item()
            #mse_loss = F.mse_loss(fake, real,  reduction='mean')
            #total_mse_loss += mse_loss.item()* len(fake)
              

        # Compute the average MSE loss over the validation set
        #average_mse_loss = total_mse_loss / total_samples
        average_mse_loss = total_mse_loss / len(val_loader.dataset)

        #mean_avg_mse, avg_mse_error = cal_epoch_rate(total_mse)

        average_mae, mae_error =  cal_epoch_rate(total_maes)
        
        #print("Average MSE Loss (reduction = sum) on Validation Set: {:.2f}".format(average_mse_loss))
        #print("Average delta on Validation Set: {:.2f}".format(average_delta))
    #return mean_passing_rate1, margin_of_error1, mean_passing_rate3, margin_of_error3, mean_avg_mse, avg_mse_error
    return average_mse_loss, average_mae
###############################End of: def val_step ####################################################
 