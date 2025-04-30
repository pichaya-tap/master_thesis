import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Critic3d(nn.Module):
    def __init__(self):
        super(Critic3d, self).__init__()

        self.conv_stack = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, stride =2, padding =1, bias=False),
            nn.SiLU(), #nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 128, kernel_size=3, stride =2, padding =1, bias=False),
            nn.BatchNorm3d(128),
            nn.SiLU(), #nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(128, 256, kernel_size=3, stride =2, padding =1, bias=False),
            nn.BatchNorm3d(256),
            nn.SiLU(), #nn.LeakyReLU(0.2, inplace=True),

            # Final convolution layer to reduce channel to 1
            nn.Conv3d(256, 1, kernel_size=(1,1,8), stride=1, padding=0, bias=False),
            nn.Flatten(),

            # Final linear layer
            nn.Linear(100, 1)
        )

    def forward(self, x, conditional_input):
        x = torch.cat([x, conditional_input], dim=1)
        x = self.conv_stack(x)      
        return x
       
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_c)
        self.swish1 = nn.SiLU()  
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_c)
        self.swish2 = nn.SiLU()  
        self.dropout = nn.Dropout3d(p=0.15)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.swish1(x)  
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.swish2(x)  
        x = self.dropout(x)
        return x


class Encoder(nn.Module):
    def __init__(self, channels, noise_dim=100):
        super().__init__()
        self.noise_dim = noise_dim
        self.enc_block = nn.ModuleList(
            [Block(channels[i], channels[i+1])
                for i in range(len(channels) - 1)])
        self.pool = nn.MaxPool3d((2,2, 2))
        # Define a scaling factor for the noise
        self.noise_scale = 0.01 

    def forward(self, x):
        # initialize an empty list to store the intermediate outputs
        block_outputs = []

        # loop through the encoder blocks
        for block in self.enc_block:
			# pass the inputs through the current encoder block, store
			# the outputs, and then apply maxpooling on the output
            x = block(x)
            #print('x block',x.shape)
            block_outputs.append(x)
            x = self.pool(x)
        #print('block_outputs length', len(block_outputs))
        # Generate noise and concatenate it with the bottleneck features
        noise = torch.randn(x.shape[0], self.noise_dim, *x.shape[2:], device=x.device)* self.noise_scale
        x = torch.cat([x, noise], dim=1)
		# return the list containing the intermediate outputs
        return x, block_outputs


class Decoder(nn.Module):
    def __init__(self, channels, noise_dim=100):
        super().__init__()
        # Add 100 to the first dimension of decChannels and convert to a list
        self.channels = [channels[0] + noise_dim] + list(channels[1:])# [228,64,32,16] 
        self.up = nn.ModuleList(
			[nn.ConvTranspose3d(self.channels[i], self.channels[i + 1], kernel_size=2, stride=2, padding=0)
			 	for i in range(len(self.channels) - 1)])
        self.dec_blocks = nn.ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
    def forward(self, x, encFeatures):
		# loop through the number of channels
        for i in range(len(self.channels) - 1):
			# pass the inputs through the upsampler blocks
            x = self.up[i](x)
            #print('x',x.shape)
			# crop the current features from the encoder blocks,
            encFeat = self.crop(encFeatures[i], x)
            #print('encFeat',encFeat.shape)
			# concatenate them with the current upsampled features,
            x = torch.cat([x, encFeat], dim=1)
            # x.shape = [32,64,4,4,32]
			# and pass the concatenated output through the current decoder block
            x = self.dec_blocks[i](x)
            #print('final decoder output ',x.shape)
		# return the final decoder output
        return x
    
    def crop(self, encFeatures, x):
        # Grab the dimensions of the inputs and crop the encoder features
        # to match the spatial dimensions of x (H, W).
        (_, _, D, H, W) = x.shape

        # crop the center region, calculate the starting indices for cropping as follows:
        start_d = (encFeatures.shape[2] - D) // 2
        start_h = (encFeatures.shape[3] - H) // 2
        start_w = (encFeatures.shape[4] - W) // 2

        # Crop the encoder features using PyTorch's slicing operations.
        cropped_features = encFeatures[:, :, start_d:start_d+D, start_h:start_h+H, start_w:start_w+W]

        # Return the cropped features
        return cropped_features




class Generator(nn.Module):
    def __init__(self, encChannels=(2, 16, 32, 64,128), decChannels=(128,64, 32, 16), nbClasses=1, retainDim=True, outSize=(16,16,256)):
        super().__init__()
        # initialize the encoder and decoder
        self.encoder = Encoder(encChannels, noise_dim=100)
        self.decoder = Decoder(decChannels, noise_dim=100)
        # initialize the regression head for 3D output
        self.head = nn.Conv3d(decChannels[-1], nbClasses, 1)
        self.retainDim = retainDim
        self.outSize = outSize

    def forward(self, x):
        # grab the features from the encoder
        b, enc_features = self.encoder(x)
        #print('after encoder b ',b.shape)
        #print('after encoder enc_features ', enc_features[::-1][0].shape)
        dec_features = self.decoder(b, enc_features[::-1][1:])
        output = self.head(dec_features)
        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them
        if self.retainDim:
            output = F.interpolate(output, self.outSize, mode='trilinear', align_corners=False)
        # Apply sigmoid activation to constrain values between 0 and 1
        output = torch.sigmoid(output)
        return output
'''
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.normal_(m.weight.data, 0.0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
'''            
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            # Xavier/Glorot Initialization
            nn.init.xavier_normal_(m.weight.data)  # or nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm3d):
            # Initialize the weights and biases for BatchNorm layers
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


def gradient_penalty(critic, real, fake, cond, device):
    """Gradient Penalty to regularize and stabilize the weight updates
    
    Args:
        critic: A PyTorch critic model to be trained.
        real: A batch of train dataset 
        fake: A batch of output from generator 
        cond: Respective conditional input of the data batch 
        device: A target device to compute on (e.g. "cuda" or "cpu")
    
    Returns:
        gradient_penalty term
    """
    # Random weight term for interpolation between real and fake samples
    epsilon = torch.rand((real.shape[0], 1, 1, 1, 1)).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (epsilon * real + ((1 - epsilon) * fake)).requires_grad_(True)

    # Calculate gradients
    mixed_scores = critic(interpolates, cond)

    #fake = Variable(torch.Tensor(real.shape[0], 1).to(device).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        inputs=interpolates,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores), #fake
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0] # get first element

    gradients = gradients.view(len(gradients), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
    #slopes  = torch.sqrt(torch.sum(gradient ** 2, dim=[1, 2, 3, 4]) + 1e-6) #small eps value to avoid Nan in sqr(0)
    #return torch.mean((slopes - 1)**2) # gradient_penalty

# To test and see summary of models # uncomment below
'''
import torchinfo    
from torchinfo import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = Generator().to(device)
initialize_weights(gen)
critic = Critic3d().to(device)
initialize_weights(critic)

summary(gen, input_size=[(32,2,16,16,256)]) # do a test pass through of an example input size 
summary(critic, input_size=[(32,1,16,16,256),(32,2,16,16,256)])
'''