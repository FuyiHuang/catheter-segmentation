import torch
import torch.nn as nn
# from BB_Unet import BBConv
from networks.BB_Unet import BBConv
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
            # nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            # nn.BatchNorm2d(ch_out),
            # nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class AttU_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1, no_grad=False, BB_boxes = 1):
        super(AttU_Net,self).__init__()

        if no_grad is True:
            no_grad_state = True
        else:
            no_grad_state = False
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)

        
        
        self.b1 = BBConv(BB_boxes, 64, 1, no_grad_state)
        self.b2 = BBConv(BB_boxes, 128, 2, no_grad_state)
        self.b3 = BBConv(BB_boxes, 256, 4, no_grad_state)
        self.b4 = BBConv(BB_boxes, 512, 8, no_grad_state)
        self.b5 = BBConv(BB_boxes, 1024, 16, no_grad_state)


    def forward(self,x,x_hessian, bb, comment= 'tr'):

        shape = x_hessian*bb
        # encoding path
        x1 = self.Conv1(x) # [4,64,256,256]

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2) # [4, 128, 128, 128]
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3) # [4, 256, 64, 64]

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4) # [4, 512, 32, 32]

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5) # [4, 1024, 16, 16]

        

        if comment == 'tr':
            s1 = self.b1(shape)
            s2 = self.b2(shape)
            s3 = self.b3(shape)
            s4 = self.b4(shape) 
            s5 = self.b5(shape)

            x1_s = x1*s1
            x2_s = x2*s2
            x3_s = x3*s3
            x4_s = x4*s4
            x5_s = x5*s5
        elif comment == 'val':
            x1_s = x1
            x2_s = x2
            x3_s = x3
            x4_s = x4
            x5_s = x5

        # decoding + concat path
        d5 = self.Up5(x5_s)
        x4 = self.Att5(g=d5,x=x4_s)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3_s)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2_s)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1_s)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return torch.sigmoid(d1)