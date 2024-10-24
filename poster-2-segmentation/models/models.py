import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms.functional as tf 

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


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, padding):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding),
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_op(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, padding):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, padding)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        return down, p


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, padding):
        self.padding = padding
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, padding)

    def forward(self, x1, x2):
        x1 = self.up(x1)


        if x1.size() != x2.size():
            x2 = tf.center_crop(x2, [x1.size(2), x1.size(3)])

        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)




class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, padding=0):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64, padding)
        self.down_convolution_2 = DownSample(64, 128, padding)
        self.down_convolution_3 = DownSample(128, 256, padding)
        self.down_convolution_4 = DownSample(256, 512, padding)

        self.bottle_neck = DoubleConv(512, 1024, padding)

        self.up_convolution_1 = UpSample(1024, 512, padding)
        self.up_convolution_2 = UpSample(512, 256, padding)
        self.up_convolution_3 = UpSample(256, 128, padding)
        self.up_convolution_4 = UpSample(128, 64, padding)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)
        
        b = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return out


if __name__ == "__main__":
    model_0_pad = UNet(in_channels=3, num_classes=1)
    x = torch.randn((1, 3, 350, 350))
    print(f"Model with 0 padding: {model_0_pad(x).shape}")
    #assert model_0_pad(x).shape == torch.Size([1, 1, 68, 68])


    model_1_pad = UNet(in_channels=3, num_classes=1, padding=1)
    x = torch.randn((1, 3, 256, 256))
    print(f"Model with 1 padding: {model_1_pad(x).shape}")
    #assert model_1_pad(x).shape == torch.Size([1, 1, 256, 256])