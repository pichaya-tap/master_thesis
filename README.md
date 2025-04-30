# PROJECT DESCRIPTION  

Accelerating dose calculations in carbon ion therapy using a Wasserstein GAN
Carbon ion beam therapy is a type of radiation therapy that uses beams of carbon ions to treat 
cancer. Its unique properties make it attractive for cancer treatment allowing for precise targeting of 
the tumor and effective treatment for tumors that are resistant to other types of radiation. Dose 
distribution prediction is currently done in computerized treatment planning systems that use 
approximations, whereas a more accurate determination of the delivered radiation dose could be 
achieved using time-consuming Monte Carlo (MC) simulations. Recently, deep learning algorithms 
have been used for fast and yet accurate prediction of the dose distribution. The improvement in 
speed of dose calculation can lead to better treatment planning and better outcomes for patients.

# GOAL

To develop, train and test a WGAN network for faster calculation of dose maps in treatment plans of 
carbon ion cancer therapy using DICOM images and Monte Carlo (MC) simulated dose maps.

# LITERATURE

Mentzel et al. , Fast and accurate dose predictions for novel radiotherapy treatments in 
heterogeneous phantoms using conditional 3D-UNet generative adversarial networks
https://arxiv.org/abs/2202.07077

# METHOD

Simulated dataset is already available from MC simulation and digital head phantoms, which are  
    • 3D matrix of the energy deposition of the carbon ion beam in a head phantom.  
    • 3D matrix of the energy deposition of the carbon ion beam in a water phantom.  
    • 3D representation of a material density matrix of phantom  
       
Investigating WGAN model training strategies and optimization of the network.
Use delta index and passing rate (1%) for model’s performance measurement of the prediction quality 

# MODEL ARCHITECTURE

The Generator network incorporates a 3D-UNet architecture and a conditional element to generate 3D images.  
It is composed of  
	- Encoder the 3D images with multiple strided 3D convolutions into low-dimensional representations  
  	- Concatenated with 100 values of random noise drawn from a Gaussian distribution  
   	- Decoder with 3D up-samplings with subsequent convolutions  
     	- Skip connections to pass information from the same level of the encoder to the same level of the decoder  
        
The generator network takes both a random noise vector and a conditional input. 
Each convolutional layers consists of 64 ﬁlters of variable size, activated using the Swish function,
stabilized using batch normalization, regularized using dropout with a rate of 15%
Output = 3D matrix of energy depositions of specific size  


The Critic network structure is simple 3D convolutional network. It receives both the generated images and the corresponding conditional input. The three input matrices are concatenated and passed through 6 transposed 3D convolutions, 
activated using the Swish function, regularized using dropout with a rate of 15%
The last single linear unit gives a continuous and quantitative rating distinguishing simulated from generated samples 


