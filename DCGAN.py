import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import multiprocessing
multiprocessing.freeze_support()

#Setting random seed for reproducibility
manualSeed = 999
print("Random seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)

#Defining some inputs for the run
#Root directory for dataset
dataroot = r"C:\Users\phenr\Downloads\celeba"

#Number of workers for dataloader
workers = 2

#Batch size during training
batch_size = 128

#Spatial size of training images. All imager will be realized to this using a transformer
image_size = 64

#Number of channels in the training images. For color images this is 3
nc = 3

#Size of z latent vector (size of generator input)
nz = 100

#Size of feature maps in discriminator 
ndf = 64

#NUmber of training epochs
num_epochs = 5

#Learning rate for optimizers
lr = 0.0002

#Beta1 hyperparameters for Adam optimizers
beta1 = 0.5

#NUmber of GPUs available 
ngpu = 1

#We can use an image folder dataset the ray we have it steup
#Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))

#Create the dataloader 
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=0)

#Decide which davice we want to run on
device = torch.device("cuda:0"if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()