import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    
    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = F.relu(self.conv2(t))
        return t

class UNet2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_block1 = Block(1, 64)
        self.down_block2 = Block(64, 128)
        self.down_block3 = Block(128, 256)
        self.down_block4 = Block(256, 512)

        self.maxpool = nn.MaxPool2d(2)

        self.bottle = Block(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.up_block1 = Block(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.up_block2 = Block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up_block3 = Block(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up_block4 = Block(128, 64)

        self.out = nn.Conv2d(64, 2, 1)

    def forward(self, t):
        down1 = self.down_block1(t)
        t = self.maxpool(down1)

        down2 = self.down_block2(t)
        t = self.maxpool(down2)

        down3 = self.down_block3(t)
        t = self.maxpool(down3)

        down4 = self.down_block4(t)
        t = self.maxpool(down4)
        t = F.dropout(t)

        t = self.bottle(t)
        t = F.dropout(t)

        t = self.up1(t)
        t = torch.cat([t, self.crop(t, down4)], 1)
        t = self.up_block1(t)
        
        t = self.up2(t)
        t = torch.cat([t, self.crop(t, down3)], 1)
        t = self.up_block2(t)

        t = self.up3(t)
        t = torch.cat([t, self.crop(t, down2)], 1)
        t = self.up_block3(t)

        t = self.up4(t)
        t = torch.cat([t, self.crop(t, down1)], 1)
        t = self.up_block4(t)

        t = self.out(t)
        return t
    
    @staticmethod
    def crop(t, d):
        pad_x1 = int(math.floor((t.shape[3] - d.shape[3]) / 2))
        pad_x2 = int(math.ceil((t.shape[3] - d.shape[3]) / 2))
        pad_y1 = int(math.floor((t.shape[2] - d.shape[2]) / 2))
        pad_y2 = int(math.ceil((t.shape[2] - d.shape[2]) / 2))
        return F.pad(d, (pad_x1, pad_x2, pad_y1, pad_y2))