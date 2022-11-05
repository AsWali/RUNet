# -*- coding: utf-8 -*-
"""
@author: Asror Wali
"""

import argparse
from multiprocessing.pool import RUN
import torch.nn as nn
import numpy as np
import torch
from torchvision import datasets, transforms

from PIL import Image
import RUNet
from vgg_perceptual_loss import VGGPerceptualLoss
from dataset import MyCustomDataset
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation with WNet')
parser.add_argument('--model', metavar='C', default="model", type=str, 
                    help='name of the saved model')
parser.add_argument('--image', metavar='C', default=None, type=str, 
                    help='path to the image')
parser.add_argument('--squeeze', metavar='K', default=4, type=int, 
                    help='Depth of squeeze layer')

blur = transforms.GaussianBlur(5, 1)

def show_image(image):
    img = image.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.show()

def pre_process_image(X, training = True, device = torch.device("cpu")):
    if training:
        downsample = nn.functional.interpolate(X, scale_factor=1/2, mode='nearest')
        input = blur(downsample).to(device)
    else:
        input = X
    upsample = nn.Upsample(scale_factor=2, mode="nearest")
    return upsample(input)

def main():
    args = parser.parse_args()
    model = RUNet.RUNet()

    model.load_state_dict(torch.load(args.model))
    model.eval()

    transform = transforms.Compose([transforms.Resize((64, 64)),
                                transforms.ToTensor()])

    image = Image.open(args.image).convert('RGB')
    x = transform(image)[None, :, :, :]

    enc = model(pre_process_image( x , training = False))
    show_image(enc[0].detach().cpu())

if __name__ == '__main__':
    main()

# PYTORCH_ENABLE_MPS_FALLBACK=1 python test.py --model "model_20221105111801" --image "images/I/8049.png"