import torch
import torch.nn as nn
from torch.nn import init


class entry_flow(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential()


class conv(nn.Module):
    def __init__(self, name_scope, in_channels, out_channels, kernel_size, stride=1, groups=1, padding=0):
        super().__init__()
        

        if 'xception' in name_scope:
            init_std = 0.09
        elif "logit" in name_scope:
            init_std = 0.01
        elif name_scope.endswith('depthwise/'):
            init_std = 0.33
        else:
            init_std = 0.06
        ## TODO 没有加正则化

        self.net = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        init.normal(self.net.weight, mean=0.0, std=init_std)
        init.constant(self.net.bias, 0.0)
    
    def forward(self, inputs):
        self.net(inputs)


class bn_relu(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        self.net = nn.Sequential()
        self.net.add_module('bn', nn.BatchNorm2d(out_channels))
        self.net.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, inputs):
        return self.net(inputs)

class seperate_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, name_scope, dilation=1, act=None):
        super().__init__()

        self.name_scope = name_scope
        self.net = nn.Sequential()

        self.name_scope = self.name_scope + 'depthwise/'
        self.net.add_module('conv1', conv(self.name_scope, in_channels, in_channels, kernel_size, 
                            stride=stride, groups=in_channels, padding=(kernel_size // 2) * dilation))
        self.name_scope = self.name_scope + 'BatchNorm/'
        self.net.add_module('bn', nn.BatchNorm2d(in_channels, eps=1e-3, momentum=0.99))
        if act == 'relu':
            self.net.add_module('relu', nn.ReLU(inplace=True))
        self.name_scope = name_scope
        self.name_scope = self.name_scope + 'pointwise/'
        self.net.add_module('conv2', conv(self.name_scope, in_channels, out_channels, 1))
        self.name_scope = self.name_scope + 'BatchNorm/'
        self.net.add_module('bn', nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.99))
        if act == 'relu':
            self.net.add_module('relu', nn.ReLU(inplace=True))
    def forward(self, inputs):
        return self.net(inputs)

