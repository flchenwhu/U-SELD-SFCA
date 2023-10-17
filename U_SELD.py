from itertools import repeat

from fvcore.nn import FlopCountAnalysis, parameter_count_table
import torch
import torch.nn as nn
from torch.nn import MaxPool2d, init, LayerNorm, AvgPool2d, Conv2d
from SFCA import SFCA

# U-SELD
class U_SELD(nn.Module):
    def __init__(self):
        super(U_SELD, self).__init__()
        self.fe = Ufe(7)
        self.adapt_maxpooling = nn.AdaptiveMaxPool2d((50, 2))
        self.bigru = nn.GRU(input_size=128, hidden_size=128, num_layers=2,
                         dropout=0.05, bidirectional=True, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(128, 128),
            nn.Linear(128, 39)
        )

    def forward(self, input):
        output = self.fe(input)
        output = output.transpose(1, 2).contiguous()
        x = output.view(output.shape[0], output.shape[1], -1).contiguous()
        (x, _) = self.bigru(x)
        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]
        # x = x[:, :, x.shape[-1]//2:] + x[:, :, :x.shape[-1]//2]
        # x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear(x))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Ufe(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super(Ufe, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.down1 = Down(out_channels)
        self.down2 = Down(2*out_channels)
        self.down3 = Down(4*out_channels)


        self.up1 = Up(8*out_channels, up_size=(10, 16))
        self.conv2 = ConvBlock(8*out_channels, 4*out_channels)
        self.up2 = Up(4*out_channels, up_size=(50, 32))
        self.conv3 = ConvBlock(4*out_channels, 2*out_channels)
        self.up3 = Up(2*out_channels, up_size=(250, 64))
        self.conv4 = ConvBlock(2*out_channels, out_channels)



    def forward(self, input):
        res1 = self.conv1(input)
        res2 = self.down1(res1)
        res3 = self.down2(res2)
        res4 = self.down3(res3)
        f3 = self.up1(res4)
        f3 = torch.cat([res3, f3], dim=1)
        f3 = self.conv2(f3)
        f2 = self.up2(f3)
        f2 = torch.cat([res2, f2], dim=1)
        f2 = self.conv3(f2)
        f1 = self.up3(f2)
        f1 = torch.cat([res1, f1], dim=1)
        f1 = self.conv4(f1)

        return f1



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block1 = nn.Sequential(
            SFCA(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, input):
        output = self.block1(input)
        return output


class Down(nn.Module):
    def __init__(self, in_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            MaxPool2d((5, 2), stride=(5, 2)),
            ConvBlock(in_channels, 2*in_channels)
        )

    def forward(self, input):
        output = self.down(input)
        return output

class Up(nn.Module):
    def __init__(self, in_channels, up_size=(10, 16), ct=False):
        super(Up, self).__init__()
        self.up = nn.UpsamplingBilinear2d(size=up_size)
        self.conv1 = ConvBlock(in_channels, in_channels // 2)


    def forward(self, input):
        output = self.up(input)
        output = self.conv1(output)
        return output








