import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer

bn_momentum = 0.1
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride
        

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width, momentum=bn_momentum)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width, momentum=bn_momentum)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# ANN
class ResNet(nn.Module):
    def __init__(self, T, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
		)

        self.fc_layers = nn.Sequential(
            nn.Linear(900, 512),
            nn.ReLU(),
		)
        
        self.linear_outputs = None



    def _forward_impl(self, x):  # torch.Size([16, 1, 28, 200])

        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers[:2](x)
        self.linear_outputs = x.clone().detach()
        x = self.fc_layers[2:](x)
        x = x.view(x.size(0), -1, 1, 1)

        return x  # torch.Size([16, 2048, 1, 1])

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(time_window, arch, block, layers, pretrained):
    model = ResNet(time_window, block, layers)  # Bottleneck, [3, 4, 23, 3]
    return model


def get_resnet(time_window, name, momentumn, pretrained=False):
    global bn_momentum
    bn_momentum = momentumn
    if name == "resnet18":
        model = _resnet(time_window, 'resnet18', BasicBlock, [2, 2, 2, 2], pretrained)
    elif name == "resnet34":
        model = _resnet(time_window, 'resnet34', BasicBlock, [3, 4, 6, 3], pretrained)
    elif name == "resnet50":
        model = _resnet(time_window, 'resnet50', Bottleneck, [3, 4, 6, 3], pretrained)
    elif name == "resnet101":
        model = _resnet(time_window, "resnet101", Bottleneck, [3, 4, 23, 3], pretrained)
    else:
        raise NotImplementedError("model {} not implemented".format(name))
    return model
