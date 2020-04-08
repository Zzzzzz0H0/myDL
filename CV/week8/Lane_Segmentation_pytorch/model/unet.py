import torch
import torch.nn as nn
# 使用reset101
from model.network import ResNet101v2
from model.module import Block

#一下采样方法
# conv-bn-relu
class UNetConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1))
        # block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_chans))
        block.append(nn.ReLU())

        block.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1))
        # block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_chans))
        block.append(nn.ReLU())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

#上采样的方法
class UNetUpBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(UNetUpBlock, self).__init__()
        # 双线性插值,采样倍数为2，与转置卷积的区别在于不会没有学习参数
        # 然后再利用1*1卷积调整channel
        self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_chans, out_chans, kernel_size=1))
        self.conv_block = UNetConvBlock(in_chans, out_chans)

    #对feature map进行裁剪
    def center_crop(self, layer, target_size):
        # 进行cat之前先进行裁剪,将layer裁剪成跟本层feacher map一致的size
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        # x: -1, bridge -2
        # up ->cat -> conv
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)
        return out


class ResNetUNet(nn.Module):
    def __init__(
        self,
        config
    ):
        super(ResNetUNet, self).__init__()
        self.n_classes = config.NUM_CLASSES
        self.encode = ResNet101v2()
        prev_channels = 2048
        # up_path的通道数[2048, 1024, 512]
        self.up_path = nn.ModuleList()
        for i in range(3):
            self.up_path.append(UNetUpBlock(prev_channels, prev_channels // 2))
            prev_channels //= 2

        # 512 -> 32 -> 16 -> n_class
        self.cls_conv_block1 = Block(prev_channels, 32)
        self.cls_conv_block2 = Block(32, 16)
        self.last = nn.Conv2d(16, self.n_classes, kernel_size=1)
        
        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_size = x.size()[2:]
        blocks = self.encode(x)
        x = blocks[-1]
        # [-1,-2],[-2,-3],[-3,-4]
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 2])
        # 将x上采样到和input_size一致
        x = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)(x)
        x = self.cls_conv_block1(x)
        x = self.cls_conv_block2(x)
        x = self.last(x)
        return x
