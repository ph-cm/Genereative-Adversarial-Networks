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

#Size of feature maps in generator
ngf = 64

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

#Implementation

#Weight Initialization
#custom weights initialization called on 'netG' and 'netD'
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#generator
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            #input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            #state size  4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            #state size 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            #state size 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            #state size 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            
            #state zise 64 x 64
        )
    def forward(self, input):
        return self.main(input)
    
#Create the generator 
netG = Generator(ngpu).to(device)

#Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

#Apply the 'weights_init' function to randomly initialize all weights
# to 'mean=0', 'stdev=0.02'
netG.apply(weights_init)

#Print Model
print(netG)

#Discriminator
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            #input is 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            #state size 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            #state size 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            #state size 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            #state size 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self,input):
        return self.main(input)
    
#Create the Discriminator
netD = Discriminator(ngpu).to(device)

#Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

#Apply the weights_init to randomly initialize all weight
netD.apply(weights_init)

#Print the model
print(netD)
    
#Loss Functions and Optimizers

#Initialize the BCELoss function
criterion = nn.BCELoss()

#Create batch of latent vectors that will use to visualize the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

#Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

#Setup Adam optimizers for both Generator and Discriminator
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

#Training

#Train the Discriminator
#Train the Generator

#Training Loop
#Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Trining Loop")

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        #(1) Update D network: maximize log(d(x)) + log(1 - D(G(z)))
        #Train with all-real batch
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        
        output = netD(real_cpu).view(-1)
        
        errD_real = criterion(output, label)
        
        errD_real.backward()
        D_x = output.mean().item()
        
        #Train with all-fake batch
        #Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        #Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        #Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        #Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        #Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        #update D
        optimizerD.step()
        
        #(2) Update G network: maximize log(D(G(z)))
        netG.zero_grad()
        label.fill_(real_label)
        #Perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        #Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        #Update G7
        optimizerG.step()
        
        #Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (epoch, num_epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        #Saving Losses for plotting later 
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        #Checking how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        iters += 1
        
#Results
plt.figure(figsize=(10, 5))
plt.title("Generator and discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

#Visualization of G's progression
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())

#Real Images vs Fake images
#Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

#Plot real image
plt.figure(figsize=(15,15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1,2,0)))

#Plot the fake images from the last epoch
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()