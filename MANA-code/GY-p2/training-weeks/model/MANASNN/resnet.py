import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer

bn_momentum = 0.1

# SNN
class ResNet(nn.Module):
    def __init__(self, T, block, layers, dropout=0):
        super(ResNet, self).__init__()

        self.T = T

        self.conv_layers__nn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1),
        )
        
        self.conv_layers = nn.Sequential(
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.BatchNorm2d(num_features=2),
            layer.AvgPool2d(kernel_size=2, stride=2),
        )
        self.conv_outputs = None

        self.fc_layers = nn.Sequential(
            layer.Linear(7200, 512),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
        )
        self.linear_outputs = None

        functional.set_step_mode(self, step_mode='m')


    def _forward_impl(self, x):

        x = self.conv_layers__nn(x)
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x_seq = self.conv_layers(x_seq)
        self.conv_outputs = x_seq.clone().detach().mean(0)      # layer_7200
        x_seq = x_seq.view(self.T, x.size(0), -1)
        x_seq = self.fc_layers[:2](x_seq)
        self.linear_outputs = x_seq.clone().detach().mean(0)    # layer_512
        x_seq = self.fc_layers[2:](x_seq)
        x_seq = x_seq.view(self.T, x.size(0), -1, 1, 1)
        fr = x_seq.mean(0)

        return fr  # torch.Size([16, 512, 1, 1])

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(time_window, dropout, usingSimpleMADA, arch, block, layers, pretrained):
    model = ResNet(time_window, block, layers, dropout)  # Bottleneck, [3, 4, 23, 3]
    return model


def get_resnet(time_window, dropout, usingSimpleMADA, name, momentumn, pretrained=False):
    global bn_momentum
    bn_momentum = momentumn
    
    model = _resnet(time_window, dropout, usingSimpleMADA, None, None, None, None)
    return model

