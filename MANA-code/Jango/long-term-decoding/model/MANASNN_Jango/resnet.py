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

        self.conv_layers = None
        self.conv_outputs = None

        self.fc_layers = nn.Sequential(
            layer.Linear(30 * 288, 512),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
		)
        self.linear_outputs = None

        functional.set_step_mode(self, step_mode='m')

    
    def _forward_impl(self, x):                             # [B, C, S, F]
        batch_size = x.shape[0]
        sequence_length = x.shape[2]
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)   # [T, B, C, S, F]                  
        self.conv_outputs = x_seq.clone().detach().mean(0)
        x_seq = x_seq.reshape(self.T, batch_size, -1)       # [T, B, C*S*F]
        x_seq = self.fc_layers(x_seq)                       # [T, B, F]
        self.linear_outputs = x_seq.clone().detach().mean(0)

        x_seq = x_seq.view(self.T, batch_size, -1, 1, 1)    # [T, B, F, 1, 1]
        fr = x_seq.mean(0)                                  # [B, F, 1, 1]          

        return fr


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
