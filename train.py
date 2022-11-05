# -*- coding: utf-8 -*-
"""
@author: Asror Wali
"""

import argparse
from multiprocessing.pool import RUN
import torch.nn as nn
import numpy as np
import time
import datetime
import torch
from torchvision import datasets, transforms
from math import log10, sqrt

import RUNet
from vgg_perceptual_loss import VGGPerceptualLoss
from dataset import MyCustomDataset
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='RUNet')
parser.add_argument('--name', metavar='name', default=str(datetime.datetime.now().strftime('%Y%m%d%H%M%S')), type=str,
                    help='Name of model')
parser.add_argument('--in_Chans', metavar='C', default=3, type=int, 
                    help='number of input channels')
parser.add_argument('--squeeze', metavar='K', default=4, type=int, 
                    help='Depth of squeeze layer')
parser.add_argument('--out_Chans', metavar='O', default=3, type=int, 
                    help='Output Channels')
parser.add_argument('--epochs', metavar='e', default=100, type=int, 
                    help='epochs')
parser.add_argument('--input_folder', metavar='f', default=None, type=str, 
                    help='Folder of input images')
parser.add_argument('--output_folder', metavar='of', default=None, type=str, 
                    help='folder of output images')

loss = VGGPerceptualLoss().to(torch.device("mps"))
blur = transforms.GaussianBlur(5, 1)
criterion = nn.MSELoss()

# Function to calculate the PSNR
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

# Function used to push items through model and calculate losses.
def train_op(model, optimizer, input, input_HD):
    enc = model(input)

    loss=perceptual_loss(input,  enc)
    loss.backward()

    mse = criterion(enc, input)
    psnr_loss = 10 * log10(1 / mse.item())
    optimizer.step()
    optimizer.zero_grad()

    return (model, loss, psnr_loss)

# Calculate perceptual loss using the vgg_perceptual_loss.py file
def perceptual_loss(x, x_prime):
    return loss(x, x_prime)

# Pre process the image, we only add the noise while training
def pre_process_image(X, training = True, device = torch.device("cpu")):
    if training:
        downsample = nn.functional.interpolate(X, scale_factor=1/2, mode='nearest')
        input = blur(downsample).to(device)
    else:
        input = X
    upsample = nn.Upsample(scale_factor=2, mode="nearest")
    return upsample(input)

# Method used to show an image
def show_image(image):
    img = image.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.show()

def main():
    # Load the arguments
    args, unknown = parser.parse_known_args()

    # Check if CUDA is available
    MPS = torch.backends.mps.is_available()
    mps_device = torch.device("mps")
    
    losses_avg = []
    psnr_losses_avg = []

    # Training image size, target output
    img_size = (256, 256)
    # Testing image size, original image size which gets doubled
    img_size2 = (128, 128)
    runet = RUNet.RUNet()

    if(MPS):
        runet = runet.to(mps_device)

    learning_rate = 0.001
    optimizer = torch.optim.Adam(runet.parameters(), lr=learning_rate)

    transform = transforms.Compose([transforms.Resize(img_size),
                                transforms.ToTensor()])
    transform2 = transforms.Compose([transforms.Resize(img_size2),
                                transforms.ToTensor()])

    dataset = MyCustomDataset(".", transform, transform)
    dataset2 = MyCustomDataset(".", transform2, transform2)

    # Train 1 image set batch size=1 and set shuffle to False
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    dataloader_test = torch.utils.data.DataLoader(dataset2, batch_size=1, shuffle=False)

    # Run for every epoch
    for epoch in range(args.epochs):

        # At 1000 epochs divide ADAM learning rate by 10
        if (epoch > 0 and epoch % 1000 == 0):
            learning_rate = learning_rate/10
            optimizer = torch.optim.Adam(runet.parameters(), lr=learning_rate)

        # Print out every epoch:
        print("Epoch = " + str(epoch))

        # Create empty lists for losses
        losses = []
        psnr_losses = []
        start_time = time.time()

        for (idx, batch) in enumerate(dataloader):
            # Train 1 image idx > 1
            if(idx > 1): break

            # Train runet
            runet, loss, psnr_loss = train_op(runet, optimizer, pre_process_image(batch["image"].to(mps_device), device= mps_device), batch["ground_truth"].to(mps_device))

            losses.append(loss.detach())
            psnr_losses.append(psnr_loss)

        losses_avg.append(torch.mean(torch.FloatTensor(losses)))
        psnr_losses_avg.append(torch.mean(torch.FloatTensor(psnr_losses)))
        print("--- %s seconds ---" % (time.time() - start_time))

    images = next(iter(dataloader_test))


    enc = runet(pre_process_image(images["image"].to(mps_device), training = False))
    show_image(enc[0].detach().cpu())
    #torch.save(runet.state_dict(), "model_" + args.name)
    #np.save("losses_" + args.name, losses_avg)
    #np.save("psnr_losses_" + args.name, psnr_losses_avg)
    print(psnr_losses_avg)
    print("Done")

if __name__ == '__main__':
    main()


# PYTORCH_ENABLE_MPS_FALLBACK=1 python3 train.py --e 1 --input_folder="images/" 