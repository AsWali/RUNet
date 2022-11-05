# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlockUP(nn.Module):
    def __init__(self, in_filters, out_filters, k, p):
        super(ResidualBlockUP, self).__init__()

        self.batchnorm1=nn.BatchNorm2d(in_filters)
        self.conv1=nn.Conv2d(in_filters, out_filters, kernel_size=k, padding=p)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0) 
        self.conv2=nn.Conv2d(out_filters, out_filters, kernel_size=k, padding=p)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0) 
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0) 
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv1(self.batchnorm1(x))
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.pixel_shuffle(x)
        
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_filters, out_filters, k, p, full = True):
        super(ResidualBlock, self).__init__()

        self.full = full
        self.conv1=nn.Conv2d(in_filters, out_filters, kernel_size=k, padding=p)
        
        
        self.batchnorm1=nn.BatchNorm2d(out_filters)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0) 

        self.conv2=nn.Conv2d(out_filters, out_filters, kernel_size=k, padding=p)
        
        self.batchnorm2=nn.BatchNorm2d(out_filters)
        self.dropout2 = nn.Dropout(0) 


    def forward(self, x):
        x = self.batchnorm1(self.conv1(x)).clamp(0)
        x = self.relu1(x)
        x = self.dropout1(x)
        if(self.full):
            x = self.batchnorm2(self.conv2(x)).clamp(0)
            x = self.dropout2(x)
        
        return x

class RUNet(nn.Module):
    def __init__(self, ch_mul=64, in_chans=3):
        super(RUNet, self).__init__()
        

        ch_mul_2 = ch_mul*2
        ch_mul_4 = ch_mul*4
        ch_mul_8 = ch_mul*8

        # Layer 1 ENC
        self.enc1=ResidualBlock(in_chans, ch_mul, k = 7, p = "same", full=False)

        # Layer 2 ENC
        self.enc2_0=ResidualBlock(ch_mul, ch_mul, k = 3, p = "same")
        self.enc2_1=ResidualBlock(ch_mul, ch_mul, k = 3, p = "same")
        self.enc2_2=ResidualBlock(ch_mul, ch_mul, k = 3, p = "same")
        self.enc2_3=ResidualBlock(ch_mul, ch_mul_2, k = 3, p = "same")

        # Layer 3 ENC
        self.enc3_0=ResidualBlock(ch_mul_2, ch_mul_2, k = 3, p = "same")
        self.enc3_1=ResidualBlock(ch_mul_2, ch_mul_2, k = 3, p = "same")
        self.enc3_2=ResidualBlock(ch_mul_2, ch_mul_2, k = 3, p = "same")
        self.enc3_3=ResidualBlock(ch_mul_2, ch_mul_4, k = 3, p = "same")

        # Layer 4 ENC
        self.enc4_0=ResidualBlock(ch_mul_4, ch_mul_4, k = 3, p = "same")
        self.enc4_1=ResidualBlock(ch_mul_4, ch_mul_4, k = 3, p = "same")
        self.enc4_2=ResidualBlock(ch_mul_4, ch_mul_4, k = 3, p = "same")
        self.enc4_3=ResidualBlock(ch_mul_4, ch_mul_4, k = 3, p = "same")
        self.enc4_4=ResidualBlock(ch_mul_4, ch_mul_4, k = 3, p = "same")
        self.enc4_5=ResidualBlock(ch_mul_4, ch_mul_8, k = 3, p = "same")

        # Layer 5 ENC
        self.enc5_0=ResidualBlock(ch_mul_8, ch_mul_8, k = 3, p = "same")
        self.enc5_1=ResidualBlock(ch_mul_8, ch_mul_8, k = 3, p = "same")
        self.enc5_bn=nn.BatchNorm2d(ch_mul_8)
        self.enc5_relu = nn.ReLU()

        # Layer 6 ENC
        self.enc6_conv=nn.Conv2d(ch_mul_8, ch_mul_8*2, kernel_size=3, padding="same")
        self.enc6_relu = nn.ReLU()

        # Layer 6 DEC
        self.dec6_conv=nn.Conv2d(ch_mul_8*2, ch_mul_8, kernel_size=3, padding="same")
        self.dec6_relu = nn.ReLU()
        self.dec6_pixel_shuffle = nn.PixelShuffle(1)

        # Layer 5 DEC
        self.dec5_0=ResidualBlockUP(ch_mul_8*2, ch_mul_8, k = 3, p = "same")

        # Layer 4 DEC
        self.dec4_0=ResidualBlockUP(640, 384, k = 3, p = "same")

        # Layer 3 DEC
        self.dec3_0=ResidualBlockUP(352, 256, k = 3, p = "same")

        # Layer 2 DEC
        self.dec2_0=ResidualBlockUP(192, 96, k = 3, p = "same")

        # Layer 1 DEC
        self.dec1_conv1=nn.Conv2d(88, 99, kernel_size=3, padding="same")
        self.dec1_relu1 = nn.ReLU()
        self.dec1_conv2=nn.Conv2d(99, 99, kernel_size=3, padding="same")
        self.dec1_relu2 = nn.ReLU()
        self.dec1_conv3=nn.Conv2d(99, 3, kernel_size=1, padding="same")


        
    def forward(self, x):
         
        # Layer 1 ENC
        enc1=self.enc1(x)

        # Layer 2 ENC
        input_layer_2 = F.max_pool2d(enc1, (2, 2))
        enc2_0=self.enc2_0(input_layer_2)
        enc2_1=self.enc2_1(enc2_0 + input_layer_2)
        enc2_2=self.enc2_2(enc2_1 + enc2_0)
        enc2_3=self.enc2_3(enc2_2 + enc2_1)
        
        enc2_2_reshape = nn.Conv2d(64, 128, kernel_size=1, padding=0).to(torch.device("mps"))
        output_layer_2 = enc2_3 + enc2_2_reshape(enc2_2)

        # Layer 3 ENC
        input_layer_3 = F.max_pool2d(output_layer_2, (2, 2))
        enc3_0=self.enc3_0(input_layer_3)
        enc3_1=self.enc3_1(enc3_0 + input_layer_3)
        enc3_2=self.enc3_2(enc3_1 + enc3_0)
        enc3_3=self.enc3_3(enc3_2 + enc3_1)
        
        enc3_2_reshape = nn.Conv2d(128, 256, kernel_size=1, padding="same").to(torch.device("mps"))
        output_layer_3 = enc3_3 + enc3_2_reshape(enc3_2)

        # Layer 4 ENC
        input_layer_4 = F.max_pool2d(output_layer_3, (2, 2))
        enc4_0=self.enc4_0(input_layer_4)
        enc4_1=self.enc4_1(enc4_0 + input_layer_4)
        enc4_2=self.enc4_2(enc4_1 + enc4_0)
        enc4_3=self.enc4_3(enc4_2 + enc4_1)
        enc4_4=self.enc4_4(enc4_3 + enc4_2)
        enc4_5=self.enc4_5(enc4_4 + enc4_3)
        
        enc4_4_reshape = nn.Conv2d(256, 512, kernel_size=1, padding="same").to(torch.device("mps"))
        output_layer_4 = enc4_5 + enc4_4_reshape(enc4_4)

        # Layer 5 ENC
        input_layer_5 = F.max_pool2d(output_layer_4, (2, 2))
        enc5_0=self.enc5_0(input_layer_5)
        enc5_1=self.enc5_1(enc5_0 + input_layer_5)
        output_layer_5 = enc5_1 + enc5_0
        enc5_bn = self.enc5_bn(output_layer_5)
        enc5_relu = self.enc5_relu(enc5_bn)

        # Layer 6 ENC
        enc6_conv = self.enc6_conv(enc5_relu)
        enc6_relu = self.enc6_relu(enc6_conv)

        # Layer 6 DEC
        dec6_conv = self.dec6_conv(enc6_relu)
        dec6_relu = self.dec6_relu(dec6_conv)

        input_layer_5_dec = torch.cat([self.dec6_pixel_shuffle(dec6_relu), torch.clone(enc5_relu)], 1)

        # Layer 5 DEC
        dec5_0 = self.dec5_0(input_layer_5_dec)
        input_layer_4_dec = torch.cat([dec5_0, torch.clone(output_layer_4)], 1)

        # Layer 4 DEC
        dec4_0 = self.dec4_0(input_layer_4_dec)
        input_layer_3_dec = torch.cat([dec4_0, torch.clone(output_layer_3)], 1)

        # Layer 3 DEC
        dec3_0 = self.dec3_0(input_layer_3_dec)
        input_layer_2_dec = torch.cat([dec3_0, torch.clone(output_layer_2)], 1)

        # Layer 2 DEC
        dec2_0 = self.dec2_0(input_layer_2_dec)
        input_layer_1_dec = torch.cat([dec2_0, torch.clone(enc1)], 1)

        # Layer 1 DEC
        dec1_conv1= self.dec1_conv1(input_layer_1_dec)
        dec1_relu1 = self.dec1_relu1(dec1_conv1)
        dec1_conv2 = self.dec1_conv2(dec1_relu1)
        dec1_relu2 = self.dec1_relu2(dec1_conv2)
        dec1_conv3 = self.dec1_conv3(dec1_relu2)

        return dec1_conv3
