import torch
import numpy as np
import torch.nn as nn
from utils.DepthwiseSeparableConvolution import SeparableConv2D


## network class
class TD_UNet(torch.nn.Module):

    def __init__(self, ch_in=1, ch_out=1, init_n_feat=32):
        super(TD_UNet, self).__init__()

        n_feat = init_n_feat
        
        self.encoder1 = TD_UNet._block(ch_in, n_feat)
        self.dilated_encoder1 = TD_UNet._dilated_block(ch_in, n_feat)
        
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = TD_UNet._block(n_feat*2, n_feat*2)
        self.dilated_encoder2 = TD_UNet._dilated_block(n_feat*2, n_feat*2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = TD_UNet._block(n_feat*4, n_feat*4)
        self.dilated_encoder3 = TD_UNet._dilated_block(n_feat*4, n_feat*4)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = TD_UNet._block(n_feat*8, n_feat*8)
        self.dilated_encoder4 = TD_UNet._dilated_block(n_feat*8, n_feat*8)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = TD_UNet._block(n_feat*16, n_feat*32)

        self.upconv4 = torch.nn.ConvTranspose2d(n_feat*32, n_feat*16, kernel_size=2, stride=2)
        self.decoder4 = TD_UNet._block((n_feat*16)*2, n_feat*16)
        self.upconv3 = torch.nn.ConvTranspose2d(n_feat*16, n_feat*8, kernel_size=2, stride=2)
        self.decoder3 = TD_UNet._block((n_feat*8)*2, n_feat*8)
        self.upconv2 = torch.nn.ConvTranspose2d(n_feat*8, n_feat*4, kernel_size=2, stride=2)
        self.decoder2 = TD_UNet._block((n_feat*4)*2, n_feat*4)
        self.upconv1 = torch.nn.ConvTranspose2d(n_feat*4, n_feat*2, kernel_size=2, stride=2)
        self.decoder1 = TD_UNet._block(n_feat*4, n_feat)

        self.conv = torch.nn.Conv2d(in_channels=n_feat, out_channels=ch_out, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc_d_1 = self.dilated_encoder1(x)
        enc_f_1 = torch.cat((enc1, enc_d_1), dim=1)
        enc_p_1 = self.pool1(enc_f_1)
        
        enc2 = self.encoder2(enc_p_1)
        enc_d_2 = self.dilated_encoder2(enc_p_1)
        enc_f_2 = torch.cat((enc2, enc_d_2), dim=1)
        enc_p_2 =self.pool2(enc_f_2)      
        
        enc3 = self.encoder3(enc_p_2)
        enc_d_3 = self.dilated_encoder3(enc_p_2)
        enc_f_3 = torch.cat((enc3, enc_d_3), dim=1)         
        enc_p_3 = self.pool3(enc_f_3)
        
        enc4 = self.encoder4(enc_p_3)
        enc_d_4 = self.dilated_encoder4(enc_p_3)
        enc_f_4 = torch.cat((enc4, enc_d_4), dim=1)   

        bottleneck = self.bottleneck(self.pool4(enc_f_4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc_f_4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc_f_3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc_f_2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc_f_1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(ch_in, n_feat):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_in, out_channels=n_feat, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True))
        
    @staticmethod
    def _dilated_block(ch_in, n_feat):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_in, out_channels=n_feat, kernel_size=3, padding=2,dilation=2, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=4, dilation=4,bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True))



class TD_UNet_light(torch.nn.Module):

    def __init__(self, ch_in=1, ch_out=1, init_n_feat=16):
        super(TD_UNet_light, self).__init__()

        n_feat = init_n_feat
        
        self.encoder1 = TD_UNet_light._block(ch_in, n_feat)
        self.dilated_encoder1 = TD_UNet_light._dilated_block(ch_in, n_feat)
        
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = TD_UNet_light._block(n_feat*2, n_feat*2)
        self.dilated_encoder2 = TD_UNet_light._dilated_block(n_feat*2, n_feat*2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = TD_UNet_light._block(n_feat*4, n_feat*4)
        self.dilated_encoder3 = TD_UNet_light._dilated_block(n_feat*4, n_feat*4)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = TD_UNet_light._block(n_feat*8, n_feat*8)
        self.dilated_encoder4 = TD_UNet_light._dilated_block(n_feat*8, n_feat*8)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = TD_UNet_light._block(n_feat*16, n_feat*32)

        self.upconv4 = torch.nn.ConvTranspose2d(n_feat*32, n_feat*16, kernel_size=2, stride=2)
        self.decoder4 = TD_UNet_light._block((n_feat*16)*2, n_feat*16)
        self.upconv3 = torch.nn.ConvTranspose2d(n_feat*16, n_feat*8, kernel_size=2, stride=2)
        self.decoder3 = TD_UNet_light._block((n_feat*8)*2, n_feat*8)
        self.upconv2 = torch.nn.ConvTranspose2d(n_feat*8, n_feat*4, kernel_size=2, stride=2)
        self.decoder2 = TD_UNet_light._block((n_feat*4)*2, n_feat*4)
        self.upconv1 = torch.nn.ConvTranspose2d(n_feat*4, n_feat*2, kernel_size=2, stride=2)
        self.decoder1 = TD_UNet_light._block(n_feat*4, n_feat)

        self.conv = torch.nn.Conv2d(in_channels=n_feat, out_channels=ch_out, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc_d_1 = self.dilated_encoder1(x)
        enc_f_1 = torch.cat((enc1, enc_d_1), dim=1)
        enc_p_1 = self.pool1(enc_f_1)
        
        enc2 = self.encoder2(enc_p_1)
        enc_d_2 = self.dilated_encoder2(enc_p_1)
        enc_f_2 = torch.cat((enc2, enc_d_2), dim=1)
        enc_p_2 =self.pool2(enc_f_2)      
        
        enc3 = self.encoder3(enc_p_2)
        enc_d_3 = self.dilated_encoder3(enc_p_2)
        enc_f_3 = torch.cat((enc3, enc_d_3), dim=1)         
        enc_p_3 = self.pool3(enc_f_3)
        
        enc4 = self.encoder4(enc_p_3)
        enc_d_4 = self.dilated_encoder4(enc_p_3)
        enc_f_4 = torch.cat((enc4, enc_d_4), dim=1)   

        bottleneck = self.bottleneck(self.pool4(enc_f_4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc_f_4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc_f_3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc_f_2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc_f_1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(ch_in, n_feat):
        return torch.nn.Sequential(
            SeparableConv2D(in_channels= ch_in, out_channels= n_feat, kernel_size=3, stride= 1, padding= 1, dilation=1, bias=False,padding_mode='zeros'),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True),)
            # SeparableConv2D(in_channels= n_feat, out_channels= n_feat, kernel_size=3, stride= 1, padding= 1, dilation=1, bias=False,padding_mode='zeros'),
            # torch.nn.BatchNorm2d(num_features=n_feat),
            # torch.nn.ReLU(inplace=True))



        
    @staticmethod
    def _dilated_block(ch_in, n_feat):
        return torch.nn.Sequential(
            SeparableConv2D(in_channels= ch_in, out_channels= n_feat, kernel_size= 3,stride=1,padding=2, dilation=2, bias=False,padding_mode='zeros'),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True),)
            # SeparableConv2D(in_channels= n_feat, out_channels= n_feat, kernel_size= 3,stride=1,padding=4, dilation=4, bias=False,padding_mode='zeros'),
            # torch.nn.BatchNorm2d(num_features=n_feat),
            # torch.nn.ReLU(inplace=True))

        
class UNet_1TD(torch.nn.Module):

    def __init__(self, ch_in=1, ch_out=1, init_n_feat=16):
        super(UNet_1TD, self).__init__()

        n_feat = init_n_feat
        
        self.encoder1 = UNet_1TD._block(ch_in, n_feat)
        self.dilated_encoder1 = UNet_1TD._dilated_block(ch_in, n_feat)
        
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = UNet_1TD._block(n_feat*2, n_feat*4)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = UNet_1TD._block(n_feat*4, n_feat*8)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = UNet_1TD._block(n_feat*8, n_feat*16)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet_1TD._block(n_feat*16, n_feat*32)

        self.upconv4 = torch.nn.ConvTranspose2d(n_feat*32, n_feat*16, kernel_size=2, stride=2)
        self.decoder4 = UNet_1TD._block((n_feat*16)*2, n_feat*16)
        self.upconv3 = torch.nn.ConvTranspose2d(n_feat*16, n_feat*8, kernel_size=2, stride=2)
        self.decoder3 = UNet_1TD._block((n_feat*8)*2, n_feat*8)
        self.upconv2 = torch.nn.ConvTranspose2d(n_feat*8, n_feat*4, kernel_size=2, stride=2)
        self.decoder2 = UNet_1TD._block((n_feat*4)*2, n_feat*4)
        self.upconv1 = torch.nn.ConvTranspose2d(n_feat*4, n_feat*2, kernel_size=2, stride=2)
        self.decoder1 = UNet_1TD._block(n_feat*4, n_feat)

        self.conv = torch.nn.Conv2d(in_channels=n_feat, out_channels=ch_out, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc_d_1 = self.dilated_encoder1(x)
        enc_f_1 = torch.cat((enc1, enc_d_1), dim=1)
        enc_p_1 = self.pool1(enc_f_1)
        
        enc2 = self.encoder2(enc_p_1)
        enc_p_2 =self.pool2(enc2)      
        
        enc3 = self.encoder3(enc_p_2)    
        enc_p_3 = self.pool3(enc3)
        
        enc4 = self.encoder4(enc_p_3)

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc_f_1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(ch_in, n_feat):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_in, out_channels=n_feat, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True))
        
    @staticmethod
    def _dilated_block(ch_in, n_feat):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_in, out_channels=n_feat, kernel_size=3, padding=2,dilation=2, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=4, dilation=4,bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True))
        
        
class UNet_2TD(torch.nn.Module):

    def __init__(self, ch_in=1, ch_out=1, init_n_feat=32):
        super(UNet_2TD, self).__init__()

        n_feat = init_n_feat
        
        self.encoder1 = UNet_2TD._block(ch_in, n_feat)
        self.dilated_encoder1 = UNet_2TD._dilated_block(ch_in, n_feat)
        
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = UNet_2TD._block(n_feat*2, n_feat*2)
        self.dilated_encoder2 = UNet_2TD._dilated_block(n_feat*2, n_feat*2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = UNet_2TD._block(n_feat*4, n_feat*8)

        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = UNet_2TD._block(n_feat*8, n_feat*16)

        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet_2TD._block(n_feat*16, n_feat*32)

        self.upconv4 = torch.nn.ConvTranspose2d(n_feat*32, n_feat*16, kernel_size=2, stride=2)
        self.decoder4 = UNet_2TD._block((n_feat*16)*2, n_feat*16)
        self.upconv3 = torch.nn.ConvTranspose2d(n_feat*16, n_feat*8, kernel_size=2, stride=2)
        self.decoder3 = UNet_2TD._block((n_feat*8)*2, n_feat*8)
        self.upconv2 = torch.nn.ConvTranspose2d(n_feat*8, n_feat*4, kernel_size=2, stride=2)
        self.decoder2 = UNet_2TD._block((n_feat*4)*2, n_feat*4)
        self.upconv1 = torch.nn.ConvTranspose2d(n_feat*4, n_feat*2, kernel_size=2, stride=2)
        self.decoder1 = UNet_2TD._block(n_feat*4, n_feat)

        self.conv = torch.nn.Conv2d(in_channels=n_feat, out_channels=ch_out, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc_d_1 = self.dilated_encoder1(x)
        enc_f_1 = torch.cat((enc1, enc_d_1), dim=1)
        enc_p_1 = self.pool1(enc_f_1)
        
        enc2 = self.encoder2(enc_p_1)
        enc_d_2 = self.dilated_encoder2(enc_p_1)
        enc_f_2 = torch.cat((enc2, enc_d_2), dim=1)
        enc_p_2 =self.pool2(enc_f_2)      
        
        enc3 = self.encoder3(enc_p_2)    
        enc_p_3 = self.pool3(enc3)
        
        enc4 = self.encoder4(enc_p_3)


        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc_f_2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc_f_1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(ch_in, n_feat):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_in, out_channels=n_feat, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True))
        
    @staticmethod
    def _dilated_block(ch_in, n_feat):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_in, out_channels=n_feat, kernel_size=3, padding=2,dilation=2, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=4, dilation=4,bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True))
        
        
        
class UNet_3TD(torch.nn.Module):

    def __init__(self, ch_in=1, ch_out=1, init_n_feat=32):
        super(UNet_3TD, self).__init__()

        n_feat = init_n_feat
        
        self.encoder1 = UNet_3TD._block(ch_in, n_feat)
        self.dilated_encoder1 = UNet_3TD._dilated_block(ch_in, n_feat)
        
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = UNet_3TD._block(n_feat*2, n_feat*2)
        self.dilated_encoder2 = UNet_3TD._dilated_block(n_feat*2, n_feat*2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = UNet_3TD._block(n_feat*4, n_feat*4)
        self.dilated_encoder3 = UNet_3TD._dilated_block(n_feat*4, n_feat*4)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = UNet_3TD._block(n_feat*8, n_feat*16)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet_3TD._block(n_feat*16, n_feat*32)

        self.upconv4 = torch.nn.ConvTranspose2d(n_feat*32, n_feat*16, kernel_size=2, stride=2)
        self.decoder4 = UNet_3TD._block((n_feat*16)*2, n_feat*16)
        self.upconv3 = torch.nn.ConvTranspose2d(n_feat*16, n_feat*8, kernel_size=2, stride=2)
        self.decoder3 = UNet_3TD._block((n_feat*8)*2, n_feat*8)
        self.upconv2 = torch.nn.ConvTranspose2d(n_feat*8, n_feat*4, kernel_size=2, stride=2)
        self.decoder2 = UNet_3TD._block((n_feat*4)*2, n_feat*4)
        self.upconv1 = torch.nn.ConvTranspose2d(n_feat*4, n_feat*2, kernel_size=2, stride=2)
        self.decoder1 = UNet_3TD._block(n_feat*4, n_feat)

        self.conv = torch.nn.Conv2d(in_channels=n_feat, out_channels=ch_out, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc_d_1 = self.dilated_encoder1(x)
        enc_f_1 = torch.cat((enc1, enc_d_1), dim=1)
        enc_p_1 = self.pool1(enc_f_1)
        
        enc2 = self.encoder2(enc_p_1)
        enc_d_2 = self.dilated_encoder2(enc_p_1)
        enc_f_2 = torch.cat((enc2, enc_d_2), dim=1)
        enc_p_2 =self.pool2(enc_f_2)      
        
        enc3 = self.encoder3(enc_p_2)
        enc_d_3 = self.dilated_encoder3(enc_p_2)
        enc_f_3 = torch.cat((enc3, enc_d_3), dim=1)         
        enc_p_3 = self.pool3(enc_f_3)
        
        enc4 = self.encoder4(enc_p_3)


        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc_f_3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc_f_2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc_f_1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(ch_in, n_feat):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_in, out_channels=n_feat, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True))
        
    @staticmethod
    def _dilated_block(ch_in, n_feat):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_in, out_channels=n_feat, kernel_size=3, padding=2,dilation=2, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=4, dilation=4,bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True))
        
        
class UNet(torch.nn.Module):

    def __init__(self, ch_in=1, ch_out=1, init_n_feat=32):
        super(UNet, self).__init__()

        n_feat = init_n_feat
        
        self.encoder1 = UNet._block(ch_in, n_feat*2) 
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = UNet._block(n_feat*2, n_feat*4)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = UNet._block(n_feat*4, n_feat*8)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = UNet._block(n_feat*8, n_feat*16)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(n_feat*16, n_feat*32)

        self.upconv4 = torch.nn.ConvTranspose2d(n_feat*32, n_feat*16, kernel_size=2, stride=2)
        self.decoder4 = UNet._block((n_feat*16)*2, n_feat*16)
        self.upconv3 = torch.nn.ConvTranspose2d(n_feat*16, n_feat*8, kernel_size=2, stride=2)
        self.decoder3 = UNet._block((n_feat*8)*2, n_feat*8)
        self.upconv2 = torch.nn.ConvTranspose2d(n_feat*8, n_feat*4, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((n_feat*4)*2, n_feat*4)
        self.upconv1 = torch.nn.ConvTranspose2d(n_feat*4, n_feat*2, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(n_feat*4, n_feat)

        self.conv = torch.nn.Conv2d(in_channels=n_feat, out_channels=ch_out, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc_p_1 = self.pool1(enc1)
        
        enc2 = self.encoder2(enc_p_1)
        enc_p_2 =self.pool2(enc2)      
        
        enc3 = self.encoder3(enc_p_2)    
        enc_p_3 = self.pool3(enc3)
        
        enc4 = self.encoder4(enc_p_3)

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4) 
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(ch_in, n_feat):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_in, out_channels=n_feat, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True))
        
    @staticmethod
    def _dilated_block(ch_in, n_feat):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_in, out_channels=n_feat, kernel_size=3, padding=2,dilation=2, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=4, dilation=4,bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True))
        

class UNet_Lighter(torch.nn.Module):

    def __init__(self, ch_in=1, ch_out=1, init_n_feat=16):
        super(UNet_Lighter, self).__init__()

        n_feat = init_n_feat
        
        self.encoder1 = UNet_Lighter._normal_block(ch_in, n_feat*2) 
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = UNet_Lighter._block(n_feat*2, n_feat*4)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = UNet_Lighter._block(n_feat*4, n_feat*8)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = UNet_Lighter._block(n_feat*8, n_feat*16)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet_Lighter._block(n_feat*16, n_feat*32)

        self.upconv4 = torch.nn.ConvTranspose2d(n_feat*32, n_feat*16, kernel_size=2, stride=2)
        self.decoder4 = UNet_Lighter._block((n_feat*16)*2, n_feat*16)
        self.upconv3 = torch.nn.ConvTranspose2d(n_feat*16, n_feat*8, kernel_size=2, stride=2)
        self.decoder3 = UNet_Lighter._block((n_feat*8)*2, n_feat*8)
        self.upconv2 = torch.nn.ConvTranspose2d(n_feat*8, n_feat*4, kernel_size=2, stride=2)
        self.decoder2 = UNet_Lighter._block((n_feat*4)*2, n_feat*4)
        self.upconv1 = torch.nn.ConvTranspose2d(n_feat*4, n_feat*2, kernel_size=2, stride=2)
        self.decoder1 = UNet_Lighter._block(n_feat*4, n_feat)

        self.conv = torch.nn.Conv2d(in_channels=n_feat, out_channels=ch_out, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc_p_1 = self.pool1(enc1)
        
        enc2 = self.encoder2(enc_p_1)
        enc_p_2 =self.pool2(enc2)      
        
        enc3 = self.encoder3(enc_p_2)    
        enc_p_3 = self.pool3(enc3)
        
        enc4 = self.encoder4(enc_p_3)

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4) 
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(ch_in, n_feat):
        return torch.nn.Sequential(
            SeparableConv2D(in_channels= ch_in, out_channels= n_feat, kernel_size=3, stride= 1, padding= 1, dilation=1, bias=False,padding_mode='zeros'),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True),
        )
 
    @staticmethod
    def _normal_block(ch_in, n_feat):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_in, out_channels=n_feat, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True),
        ) 
    
        
    @staticmethod
    def _dilated_block(ch_in, n_feat):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_in, out_channels=n_feat, kernel_size=3, padding=2,dilation=2, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=4, dilation=4,bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True))
        


        
    
class Dual_Branch_UNet(torch.nn.Module):

    def __init__(self, ch_in=1, ch_out=1, init_n_feat=16):
        super(Dual_Branch_UNet, self).__init__()

        n_feat = init_n_feat
        
        self.encoder1 = Dual_Branch_UNet._block(ch_in, n_feat*2) 
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = Dual_Branch_UNet._block(n_feat*2, n_feat*4)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = Dual_Branch_UNet._block(n_feat*4, n_feat*8)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = Dual_Branch_UNet._block(n_feat*8, n_feat*16)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = Dual_Branch_UNet._block(n_feat*32, n_feat*32)

        self.upconv4 = torch.nn.ConvTranspose2d(n_feat*32, n_feat*16, kernel_size=2, stride=2)
        self.decoder4 = Dual_Branch_UNet._block((n_feat*16)*2, n_feat*16)
        self.upconv3 = torch.nn.ConvTranspose2d(n_feat*16, n_feat*8, kernel_size=2, stride=2)
        self.decoder3 = Dual_Branch_UNet._block((n_feat*8)*2, n_feat*8)
        self.upconv2 = torch.nn.ConvTranspose2d(n_feat*8, n_feat*4, kernel_size=2, stride=2)
        self.decoder2 = Dual_Branch_UNet._block((n_feat*4)*2, n_feat*4)
        self.upconv1 = torch.nn.ConvTranspose2d(n_feat*4, n_feat*2, kernel_size=2, stride=2)
        self.decoder1 = Dual_Branch_UNet._block(n_feat*4, n_feat)

        self.conv = torch.nn.Conv2d(in_channels=n_feat, out_channels=ch_out, kernel_size=1)

    def forward(self, x1, x2):
        
        # branch 1 
        enc1 = self.encoder1(x1)
        enc_p_1 = self.pool1(enc1)
        
        enc2 = self.encoder2(enc_p_1)
        enc_p_2 =self.pool2(enc2)      
        
        enc3 = self.encoder3(enc_p_2)    
        enc_p_3 = self.pool3(enc3)
        
        enc4 = self.encoder4(enc_p_3)
        
        # branch 2 
        b2_enc1 = self.encoder1(x2)
        b2_enc_p_1 = self.pool1(b2_enc1)
        
        b2_enc2 = self.encoder2(b2_enc_p_1)
        b2_enc_p_2 =self.pool2(b2_enc2)      
        
        b2_enc3 = self.encoder3(b2_enc_p_2)    
        b2_enc_p_3 = self.pool3(b2_enc3)
        
        b2_enc4 = self.encoder4(b2_enc_p_3) 
        

        bottleneck = self.bottleneck(torch.cat((self.pool4(enc4), self.pool4(b2_enc4)), dim=1) )

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4) 
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(ch_in, n_feat):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_in, out_channels=n_feat, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True))
        
    @staticmethod
    def _dilated_block(ch_in, n_feat):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_in, out_channels=n_feat, kernel_size=3, padding=2,dilation=2, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=4, dilation=4,bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True))