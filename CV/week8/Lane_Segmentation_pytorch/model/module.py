import torch
import torch.nn as nn
import torch.nn.functional as F


#  最基本的Block
class Block(nn.Module):
    def __init__(self, in_ch,out_ch, kernel_size=3, padding=1, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)
        # bn层，是对一个batch输出（不是参数）做归一化
        self.bn1 = nn.BatchNorm2d(out_ch)
        # relu不需要反向求导，所以可以用inplace来节省内存
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        # 这里用的是cov->bn->relu的顺序，bn的原始论文就是这样的，
        # 先bn再relu的好处是有效的利用激活函数（对sigmod和tanh），对relu效果没那么明显
        # conv->relu->bn的说法是，该层的输出是下一层的输入，所以需要在后面进行bn
        # 实际效果需要通过训练效果判断（知乎上说先relu再bn效果更好）
        out = self.relu1(self.bn1(self.conv1(x)))
        return out

# resnet的基本block
class ResBlock(nn.Module):
    def __init__(self, in_ch,out_ch, kernel_size=3, padding=1, stride=1):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        # 这里用的是bn->relu->conv,目前看和Block没有区别，因为该层的输出是下层的输入
        # 残差处理时，没有使用relu，是为了构造更深的网络
        out = self.conv1(self.relu1(self.bn1(x)))
        return out

# resnet的中间的bottleneck,输入和输出的channel和feature map不发生变化
# kernel: 1*1->3*3->1*1
# expansion表示最后1*1卷积放大的倍数为4
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_chans, out_chans):
        super(Bottleneck, self).__init__()
        assert out_chans % 4 == 0
        # 先1*1到out channel的1/4,再3*3不改变kernel和feature map，最后1*1到out channel
        self.block1 = ResBlock(in_chans, int(out_chans / 4), kernel_size=1, padding=0)
        self.block2 = ResBlock(int(out_chans / 4), int(out_chans / 4), kernel_size=3, padding=1)
        self.block3 = ResBlock(int(out_chans / 4), out_chans, kernel_size=1, padding=0)

    def forward(self, x):
        # 没有对输入的x进行任何处理，需要保证x和out的channel和feacher map一致
        identity = x
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out += identity
        return out


# resnet每组的第一个bottleneck，对输入的x进行了feacher map的变化（1/2），stride不为1
class DownBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_chans, out_chans, stride=2):
        super(DownBottleneck, self).__init__()
        assert out_chans % 4 == 0
        self.block1 = ResBlock(in_chans, int(out_chans / 4), kernel_size=1, padding=0, stride=stride)
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=1, padding=0, stride=stride)
        self.block2 = ResBlock(int(out_chans / 4), int(out_chans / 4), kernel_size=3, padding=1)
        self.block3 = ResBlock(int(out_chans / 4), out_chans, kernel_size=1, padding=0)

    def forward(self, x):
        # 需要1*1 对x进行处理，保证和out一样的channel和feature map
        identity = self.conv1(x)
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out += identity
        return out

# 生成vgg和resnet，返回生成的module
# layer_list为每层网络的输出channel列表
def make_layers(in_channels, layer_list, name="vgg"):
    layers = []
    if name == "vgg":
        # vgg的网络都是基本的block，迭代生成即可
        for v in layer_list:
            layers += [Block(in_channels, v)]
            in_channels = v
    elif name == "resnet":
        # reset第一个是DownBottlneck,其余都是Bottleneck
        # 这里没有basicblock，所以resnet应该都是50层以上的
        layers += [DownBottleneck(in_channels, layer_list[0])]
        in_channels = layer_list[0]
        for v in layer_list[1:]:
            layers += [Bottleneck(in_channels, v)]
            in_channels = v
    return nn.Sequential(*layers)


# 抽象的vgg和resnet的接口module
class Layer(nn.Module):
    def __init__(self, in_channels, layer_list, net_name):
        super(Layer, self).__init__()
        self.layer = make_layers(in_channels, layer_list, name=net_name)

    def forward(self, x):
        # forward只需要将sequential的module调用即可
        out = self.layer(x)
        return out

# deeplabv3p的aspp部分
class ASPP(nn.Module):

    def __init__(self, in_chans, out_chans, rate=1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )
        self.branch5_avg = nn.AdaptiveAvgPool2d(1)
        self.branch5_conv = nn.Conv2d(in_chans, out_chans, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(out_chans)
        self.branch5_relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_chans * 5, out_chans, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True))

    def forward(self, x):
        b, c, h, w = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = self.branch5_avg(x)
        global_feature = self.branch5_relu(self.branch5_bn(self.branch5_conv(global_feature)))
        global_feature = F.interpolate(global_feature, (h, w), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result
