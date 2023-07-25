"""
This script defines the LightCTS model.

The main components of the script are:

1. channel_shuffle: A function that shuffles the channels of the feature maps in the L-TCN.

2. SELayer: A class defining the squeeze-and-excitation layer, which re-calibrates spatial dimension feature responses adaptively.

3. LightCTS: The main class defining the LightCTS model. It consists of an embedding module (a single convolutional layer), multiple L-TCN layers (where the channel shuffle and SE mechanisms are applied), and a GLFormer network that takes the output of the L-TCN layers, and an output module to generate the final output.

"""

# Import necessary libraries
import torch
import torch.nn as nn
from torch.nn import BatchNorm2d, Conv2d, ModuleList
import torch.nn.functional as F
# Import GLFormer as S-operator
from GLFormer_model import GLFormer

# Define a function to shuffle the channels of feature maps
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

# Define the squeeze-and-excitation layer
class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

# Define the main model of LightCTS
class LightCTS(nn.Module):
    def __init__(self, supports=None, in_dim=2, out_dim=12, hid_dim=32, nglf=6, cnn_layers=4, group=4):
        super(LightCTS, self).__init__()

        self.cnn_layers = cnn_layers
        self.Filter_Convs = ModuleList()
        self.Gate_Convs = ModuleList()
        self.group=group
        D = [1, 2, 4, 8]
        additional_scope = 1
        receptive_field = 1

        # Define the embedding module which consists of one convolutional layer
        self.Start_Conv = Conv2d(in_channels=in_dim, out_channels=hid_dim, kernel_size=(1, 1))

        # Define the L-TCN layers
        for i in range(self.cnn_layers):
            self.Filter_Convs.append(Conv2d(hid_dim, hid_dim, (1, 2), dilation=D[i], groups=group))
            self.Gate_Convs.append(Conv2d(hid_dim, hid_dim, (1, 2), dilation=D[i], groups=group))
            receptive_field += additional_scope
            additional_scope *= 2
        self.receptive_field=receptive_field
        depth = list(range(self.cnn_layers))

        # Define the GLFormer network
        self.GLFormer_Network = GLFormer(hid_dim, nglf)
        mask0 = supports[0].detach()
        mask1 = supports[1].detach()
        mask = mask0 + mask1
        self.mask = mask == 0
        self.bn = ModuleList([BatchNorm2d(hid_dim) for _ in depth])

        # Define the output module
        self.Output_Conv1 = nn.Linear(hid_dim, hid_dim*4)
        self.Output_Conv2 = nn.Linear(hid_dim*4, out_dim)

        self.se=SELayer(hid_dim)


    def forward(self, input):

        # Check if the input size is less than the receptive field
        in_len = input.size(3)
        if in_len < self.receptive_field:
            # If it is, pad the input tensor
            input = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        x = self.Start_Conv(input)
        skip = 0

        # Pass the tensor through each L-TCN layer
        for i in range(self.cnn_layers):
            residual = x
            filter = torch.tanh(self.Filter_Convs[i](residual))
            gate = torch.sigmoid(self.Gate_Convs[i](residual))
            x = filter * gate
            x = channel_shuffle (x,self.group)

            # Apply the last-shot compression
            try:
                skip += x[:, :, :, -1:]
            except:
                skip = 0
            if i == self.cnn_layers-1:
                break
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        # Squeeze the temporal dimension and apply the SE layer
        x = torch.squeeze(skip, dim=-1)
        x=self.se(x)
        x = x.transpose(1, 2)
        x_residual = x

        # Send the compressed feature maps to the GLFormer network
        x = self.GLFormer_Network(x,self.mask)
        x += x_residual

        # Pass the compressed feature through the output module
        x = F.relu(self.Output_Conv1(x))
        x = self.Output_Conv2(x)

        # Return the forecasts
        return x.transpose(1, 2).unsqueeze(-1)

