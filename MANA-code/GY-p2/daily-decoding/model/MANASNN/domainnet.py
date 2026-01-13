import torch.nn as nn
import torch.nn.functional as F
from .resnet import get_resnet,ResNet
import torch

feature_dict = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048, "our_net":1024}


class DomainNet_encoder(nn.Module):
    def __init__(self, time_window, backbone, bn_momentum, pretrained=True, data_parallel=True, dropout=0):
        super(DomainNet_encoder, self).__init__()
        self.linear_outputs = None
        encoder = get_resnet(time_window, dropout, backbone, momentumn=bn_momentum, pretrained=pretrained)
        if data_parallel:
            self.encoder = nn.DataParallel(encoder)
        else:
            self.encoder = encoder

    def forward(self, x):
        feature = self.encoder(x)
        return feature

    def get_linear_outputs(self):

        if isinstance(self.encoder, ResNet):
            return self.encoder.linear_outputs
        else:
            raise TypeError("The encoder is not an instance of ResNet")


class DomainNetClassifier(nn.Module):
    def __init__(self, backbone, classes=126, data_parallel=True):
        super(DomainNetClassifier, self).__init__()
        linear = nn.Sequential()
        # linear.add_module("fc", nn.Linear(feature_dict[backbone], classes))
        linear.add_module("fc", nn.Linear(512, classes))
        if data_parallel:
            self.linear = nn.DataParallel(linear)
        else:
            self.linear = linear

    def forward(self, feature):
        feature = torch.flatten(feature, 1)
        feature = self.linear(feature)
        return feature
