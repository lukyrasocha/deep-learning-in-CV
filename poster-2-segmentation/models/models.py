import torch
import torch.nn as nn
import torch.nn.functional as F

class EncDec(nn.Module):
    def __init__(self, input_channels=3, output_channels=1):
        super().__init__()

        # Encoder 
        self.enc_conv0 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  
        
        self.enc_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc_conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc_conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(512, 1024, kernel_size=3, padding=1)

        # Decoder 
        self.upsample0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv0 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv3 = nn.Conv2d(128, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        e0 = F.relu(self.enc_conv0(x))
        e0_pooled = self.pool0(e0)
        
        e1 = F.relu(self.enc_conv1(e0_pooled))
        e1_pooled = self.pool1(e1)
        
        e2 = F.relu(self.enc_conv2(e1_pooled))
        e2_pooled = self.pool2(e2)
        
        e3 = F.relu(self.enc_conv3(e2_pooled))
        e3_pooled = self.pool3(e3)

        # Bottleneck
        b = F.relu(self.bottleneck_conv(e3_pooled))

        # Decoder
        d0 = F.relu(self.dec_conv0(self.upsample0(b)))
        d1 = F.relu(self.dec_conv1(self.upsample1(d0)))
        d2 = F.relu(self.dec_conv2(self.upsample2(d1)))
        d3 = self.dec_conv3(self.upsample3(d2))

        return d3 # we return logits! (i.e. not probabilities)