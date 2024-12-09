import numpy as np
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    U-Net model single convolution block.
    """
    def __init__(self, in_channels, out_channels):
        """
        Encoder block of the basic U-Net model
        for musical source separation.
        
        Args:
            in_channels (int): number of input filters
            out_channels (int): number of output filters
        """
        super().__init__()

        # padding computation from
        # https://github.com/ws-choi/Conditioned-U-Net-pytorch/blob/master/models/unet_model.py
        k, s = (5, 2)
        pad = ((k - s + 1) // 2)

        layers = [
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=k, stride=s, padding=pad),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        ]
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_block(x)
        return x


class DeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, mask=False):

        super().__init__()

        # padding computation from
        # https://github.com/ws-choi/Conditioned-U-Net-pytorch/blob/master/models/unet_model.py
        k, s = (5, 2)
        o = abs(k % 2 - s % 2)
        pad = (k - s + o) // 2
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=5, stride=2,
                               padding=pad, output_padding=o),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        if dropout:
            # insert dropout layer before activation layer
            layers.insert(2, nn.Dropout(0.5))

        if mask:
            # replace the activation layer with the sigmoid function
            layers[-1] = nn.Sigmoid()

        self.deconv_block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.deconv_block(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, filters=[16, 32, 64, 128, 256]):
        """
        U-Net model for musical source separation.
        
        Args:
            in_channels (int): number of input channels
            filters (list, optional): number of filters in each block
        """
        super().__init__()

        # encoding layers
        self.down1 = ConvBlock(in_channels, filters[0])
        self.down2 = ConvBlock(filters[0], filters[1])
        self.down3 = ConvBlock(filters[1], filters[2])
        self.down4 = ConvBlock(filters[2], filters[3])
        self.down5 = ConvBlock(filters[3], filters[4])

        # bottleneck layer
        self.bottleneck = ConvBlock(filters[4], 2 * filters[4])

        # decoding layers
        # multiply input channels by 2 to account for concatenating
        # skip connections
        self.up1 = DeConvBlock(2 * filters[4], filters[4])
        self.up2 = DeConvBlock(2 * filters[4], filters[3], dropout=True)
        self.up3 = DeConvBlock(2 * filters[3], filters[2], dropout=True)
        self.up4 = DeConvBlock(2 * filters[2], filters[1], dropout=True)
        self.up5 = DeConvBlock(2 * filters[1], filters[0])
        
        # final layer to generate a soft mask
        self.map = DeConvBlock(2 * filters[0], 1, mask=True)
 
    def forward(self, x):
        # collect skip connections
        skip = []

        # save input
        input_x = x

        # downward path
        x = self.down1(x)
        skip.append(x)

        x = self.down2(x)
        skip.append(x)

        x = self.down3(x)
        skip.append(x)

        x = self.down4(x)
        skip.append(x)

        x = self.down5(x)
        skip.append(x)

        # bottleneck
        x = self.bottleneck(x)

        # upward path
        # add skip connections by concatenating on channel axis
        x = self.up1(x)

        x = torch.cat((skip[-1], x), dim=1)
        x = self.up2(x)

        x = torch.cat((skip[-2], x), dim=1)
        x = self.up3(x)
        
        x = torch.cat((skip[-3], x), dim=1)
        x = self.up4(x)

        x = torch.cat((skip[-4], x), dim=1)
        x = self.up5(x)

        x = torch.cat((skip[-5], x), dim=1)
        mask = self.map(x)  # collapse to output mask

        # mulitply input by mask to get separated source
        output_x = torch.mul(input_x, mask)

        return output_x
