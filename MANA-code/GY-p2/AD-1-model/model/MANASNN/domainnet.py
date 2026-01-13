import torch.nn as nn
import torch.nn.functional as F
from .resnet import get_resnet, ResNet
import torch

feature_dict = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048, "our_net":1024}

class DomainNet_encoder(nn.Module):
    def __init__(self, time_window, backbone, bn_momentum, pretrained=True, data_parallel=True, dropout=0, usingSimpleMADA=False):
        super(DomainNet_encoder, self).__init__()
        self.conv_outputs = None
        self.linear_outputs = None
        encoder = get_resnet(time_window, dropout, usingSimpleMADA, backbone, momentumn=bn_momentum, pretrained=pretrained)
        if data_parallel:
            self.encoder = nn.DataParallel(encoder)
        else:
            self.encoder = encoder

    def forward(self, x):
        feature = self.encoder(x)
        return feature

    def get_conv_outputs(self):
        return self.encoder.conv_outputs

    def get_linear_outputs(self):
        return self.encoder.linear_outputs


class DomainNetClassifier(nn.Module):
    def __init__(self, backbone, classes=126, data_parallel=True, usingSimpleMADA=False):
        super(DomainNetClassifier, self).__init__()
        linear = nn.Sequential()
        if usingSimpleMADA:
            linear.add_module("fc", nn.Linear(512, classes))
        else:
            linear.add_module("fc", nn.Linear(512, classes))
            
        if data_parallel:
            self.linear = nn.DataParallel(linear)
        else:
            self.linear = linear

    def forward(self, feature):
        feature = torch.flatten(feature, 1)
        feature = self.linear(feature)
        return feature



