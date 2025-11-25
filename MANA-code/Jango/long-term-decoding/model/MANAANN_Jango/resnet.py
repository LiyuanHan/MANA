import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer
import torchprofile

bn_momentum = 0.1

# ANN
class ResNet(nn.Module):
    def __init__(self, T, block, layers, dropout=0):
        super(ResNet, self).__init__()

        self.conv_layers = None
        self.conv_outputs = None

        self.fc_layers = nn.Sequential(
            nn.Linear(30 * 288, 512),
            nn.ReLU(),
		)
        self.linear_outputs = None

    
    def _forward_impl(self, x):                             # [B, C, S, F]
        batch_size = x.shape[0]
        sequence_length = x.shape[2]                     
        self.conv_outputs = x.clone().detach()
        x = x.reshape(batch_size, -1)     # [B, C*S*F]
        x = self.fc_layers(x)                               # [B, F]
        self.linear_outputs = x.clone().detach()

        x = x.view(batch_size, -1, 1, 1) # [B, F, 1, 1]

        return x


    def forward(self, x):
        return self._forward_impl(x)
    

def _resnet(time_window, dropout, usingSimpleMADA, arch, block, layers, pretrained):
    model = ResNet(time_window, block, layers, dropout)
    return model


def get_resnet(time_window, dropout, usingSimpleMADA, name, momentumn, pretrained=False):
    global bn_momentum
    bn_momentum = momentumn
    
    model = _resnet(time_window, dropout, usingSimpleMADA, None, None, None, None)
    return model
