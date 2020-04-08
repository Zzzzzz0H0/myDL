import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
# 带膨胀卷积的resnet

bn_mom = 0.0003
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
}

# 兼容膨胀卷积
def conv3x3(in_planes, out_planes, stride=1, atrous=1):
    """3x3 convolution with padding"""
    # padding=atrous ,保证在kernel为3是，膨胀卷积和普通卷积的feacher map一致
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1 * atrous, dilation=atrous, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_chans, out_chans, stride=1, atrous=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chans, out_chans, stride, atrous)
        self.bn1 = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_chans, out_chans)
        self.bn2 = nn.BatchNorm2d(out_chans)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # basic block
        # conv->bn->relu->conv->bn->downsample->relu
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_chans, out_chans, stride=1, atrous=1, downsample=None):
        # stride 为1 feacher map不变，为2，feacher map变为1/2
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chans)
        # 需要dilation参数为1是普通卷积，为其他值时是膨胀卷积
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=stride,
                               padding=1 * atrous, dilation=atrous, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chans * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # 先1 *1 ，3*3 最后1*1
        # 3*3 的时候加入膨胀卷积
        # conv后面接bn的一律可以设置bias为False
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_Atrous(nn.Module):

    def __init__(self, block, layers, atrous=None, os=16):
        """__init__

        :param block: block 类型
        :param layers: 前四层layer的block数目,5,6层block数跟第四层一致
        :param atrous: 膨胀系数
        :param os: 下采样倍数，如果为8，则
        """
        super(ResNet_Atrous, self).__init__()
        stride_list = None
        if os == 8:
            stride_list = [2, 1, 1]
        elif os == 16:
            stride_list = [2, 2, 1]
        else:
            raise ValueError('resnet_atrous.py: output stride=%d is not supported.' % os)

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
       # 2倍下采样,stride=2

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 64, layers[0])
        # 4倍下采样,此次是有maxpool 导致的，不是conv导致的
        # DRN的论文指出这里用stride 2代替pool可以防止网格化
        # todo:去掉pool

        self.layer2 = self._make_layer(block, 256, 128, layers[1], stride=stride_list[0])
        # 8倍下采样,stride=2,没有使用膨胀卷积

        self.layer3 = self._make_layer(block, 512, 256, layers[2], stride=stride_list[1], atrous=16 // os)
        # 8倍/16倍下采样,stride=1/2,如果os为8，则atros为2,stride 为1;如os为16，则atros为1，stride为2

        # 以下layer都是使用了膨胀卷积，所以stride都为1，并且需要atros
        # os 为16，膨胀系数1，2，1  保证感受野和原模型一致
        # os 为8，膨胀系数2，4，2   锯齿状有效防止网格化
        self.layer4 = self._make_layer(block, 1024, 512, layers[3], stride=stride_list[2],
                                       atrous=[item * 16 // os for item in atrous])
        self.layer5 = self._make_layer(block, 2048, 512, layers[3], stride=1, atrous=[item*16//os for item in atrous])
        self.layer6 = self._make_layer(block, 2048, 512, layers[3], stride=1, atrous=[item*16//os for item in atrous])
        self.layers = []

        for m in self.modules():
            # relu激活函数用kaiming初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_chans, out_chans, blocks, stride=1, atrous=None):
        """_make_layer

        :param block: block 类型
        :param in_chans: 输入channel 
        :param out_chans: 输出channel
        :param blocks:  block 层数
        :param stride: stride 
        :param atrous: 膨胀系数
        """
        downsample = None
        # 如果没有atrous参数的，说明为普通卷积，所以每个block都为1
        if atrous == None:
            atrous = [1] * blocks
        # 如果整数的1或者2这里，则每层值都为该值,为了兼容上面的8倍或者16倍下采样
        elif isinstance(atrous, int):
            atrous_list = [atrous] * blocks
            atrous = atrous_list
        # 剩下的都为膨胀卷积,值为1，2，1,后续也都是1

        if stride != 1 or in_chans != out_chans * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_chans, out_chans * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chans * block.expansion),
            )

        layers = []
        layers.append(block(in_chans, out_chans, stride=stride, atrous=atrous[0], downsample=downsample))
        in_chans = out_chans*4
        for i in range(1, blocks):
            layers.append(block(in_chans, out_chans, stride=1, atrous=atrous[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        layers_list = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        #  print(x.shape)
        layers_list.append(x)
        x = self.layer2(x)
        #  print(x.shape)
        layers_list.append(x)
        x = self.layer3(x)
        #  print(x.shape)
        layers_list.append(x)
        x = self.layer4(x)
        #  print(x.shape)
        x = self.layer5(x)
        #  print(x.shape)
        x = self.layer6(x)
        #  print(x.shape)
        layers_list.append(x)

        return layers_list


def resnet50_atrous(pretrained=True, os=16, **kwargs):
    """resnet50_atrous
    :param pretrained: 是否加载预训练参数
    :param os: 下采样倍数
    :param **kwargs:
    """
    """Constructs a atrous ResNet-50 model."""
    model = ResNet_Atrous(Bottleneck, [3, 4, 6, 3], atrous=[1, 2, 1], os=os, **kwargs)
    if pretrained:
        old_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
        print(1)
    return model


def resnet101_atrous(pretrained=True, os=16, **kwargs):
    """Constructs a atrous ResNet-101 model."""
    # [1,2,1]的膨胀系数可以防止网格化
    model = ResNet_Atrous(Bottleneck, [3, 4, 23, 3], atrous=[1, 2, 1], os=os, **kwargs)
    if pretrained:
        old_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model

x = torch.randn(1, 3, 224, 224)
#  m = resnet50_atrous(pretrained=False, os=8)
#  --------------------
#  torch.Size([1, 256, 56, 56])
#  torch.Size([1, 512, 28, 28])
#  torch.Size([1, 1024, 28, 28])
#  torch.Size([1, 2048, 28, 28])
#  torch.Size([1, 2048, 28, 28])
#  torch.Size([1, 2048, 28, 28])

#  m = resnet50_atrous(pretrained=False, os=16)
#  --------------------
#  torch.Size([1, 256, 56, 56])
#  torch.Size([1, 512, 28, 28])
#  torch.Size([1, 1024, 14, 14])
#  torch.Size([1, 2048, 14, 14])
#  torch.Size([1, 2048, 14, 14])
#  torch.Size([1, 2048, 14, 14])
#  m(x)
