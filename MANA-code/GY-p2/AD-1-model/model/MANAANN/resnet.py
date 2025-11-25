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

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
		)
        self.conv_outputs = None

        self.fc_layers = nn.Sequential(
            nn.Linear(7200, 512),
            nn.ReLU(),
		)
        self.linear_outputs = None

    
    def _forward_impl(self, x):
        
        x = self.conv_layers(x)
        self.conv_outputs = x.clone().detach()
        x = x.view(x.size(0), -1)
        x = self.fc_layers[:3](x)
        self.linear_outputs = x.clone().detach()
        x = self.fc_layers[3:](x)
        x = x.view(x.size(0), -1, 1, 1)

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
