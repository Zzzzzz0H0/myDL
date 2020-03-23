import torch
import torch.nn as nn
from models.deeplab_party import conv, seperate_conv, bn_relu

class deeplabv3p(nn.Module):
    def __init__(self, in_channels, label_number):
        super().__init__()

        self.name_scope = 'xception_65/entry_flow/conv1/'
        self.conv1 = conv(self.name_scope, in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.name_scope = 'xception_65/entry_flow/conv2/'
        self.conv2 = conv(self.name_scope, 32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)

        self.name_scope = 'xception_65/entry_flow/block1/separable_conv1/'
        self.relu3 = nn.ReLU(inplace=True)
        ## block1_results[0]
        self.seperate_conv1 = seperate_conv(32, 64, 3, 1, self.name_scope)
        self.name_scope = 'xception_65/entry_flow/block1/separable_conv2/'
        self.relu4 = nn.ReLU(inplace=True)
        ## block1_results[1]
        self.seperate_conv2 = seperate_conv(64, 64, 3, 1, self.name_scope)
        self.name_scope = 'xception_65/entry_flow/block1/separable_conv3/'
        self.relu5 = nn.ReLU(inplace=True)
        ## block1_results[2]
        self.seperate_conv3 = seperate_conv(64, 64, 3, 2, self.name_scope)

        self.name_scope = 'xception_65/entry_flow/block1/shortcut/'
        ## 这里用relu2输出
        self.conv3 = conv(self.name_scope, 32, 64, kernel_size=1, stride=2)
        self.bn3 = nn.BatchNorm2d(64)

        ## TODO 逐元素相加  bn3+seperate_conv3   name:addition1

        self.name_scope = 'xception_65/entry_flow/block2/separable_conv1/'
        self.relu6 = nn.ReLU(inplace=True)
        ## block2_results[0]
        self.seperate_conv4 = seperate_conv(64, 128, 3, 1, self.name_scope)
        self.name_scope = 'xception_65/entry_flow/block2/separable_conv2/'
        self.relu7 = nn.ReLU(inplace=True)
        ## block2_results[1]
        self.seperate_conv5 = seperate_conv(128, 128, 3, 1, self.name_scope)
        self.name_scope = 'xception_65/entry_flow/block2/separable_conv3/'
        self.relu8 = nn.ReLU(inplace=True)
        ## block2_results[2]
        self.seperate_conv6 = seperate_conv(128, 128, 3, 2, self.name_scope)

        self.name_scope = 'xception_65/entry_flow/block2/shortcut/'
        ## 这里用addition1输出
        self.conv4 = conv(self.name_scope, 64, 128, kernel_size=1, stride=2)
        self.bn4 = nn.BatchNorm2d(128)
        ## TODO 逐元素相加  bn4+seperate_conv6  name:addition2

        self.name_scope = 'xception_65/entry_flow/block3/separable_conv1/'
        self.relu9 = nn.ReLU(inplace=True)
        ## block3_results[0]
        self.seperate_conv7 = seperate_conv(128, 256, 3, 1, self.name_scope)
        self.name_scope = 'xception_65/entry_flow/block3/separable_conv2/'
        self.relu10 = nn.ReLU(inplace=True)
        ## block3_results[1]
        self.seperate_conv8 = seperate_conv(256, 256, 3, 1, self.name_scope)
        self.name_scope = 'xception_65/entry_flow/block3/separable_conv3/'
        self.relu11 = nn.ReLU(inplace=True)
        ## block3_results[2]
        self.seperate_conv9 = seperate_conv(256, 256, 3, 2, self.name_scope)

        self.name_scope = 'xception_65/entry_flow/block3/shortcut/'
        ## 这里用addition2输出
        self.conv5 = conv(self.name_scope, 128, 256, kernel_size=1, stride=2)
        self.bn5 = nn.BatchNorm2d(256)
        ## TODO 逐元素相加  bn5+seperate_conv9  name:addition3

        ## data=addition3  decode_shortcut1=block1_results[1]  decode_shortcut2=block2_results[2]
        self.name_scope = 'xception_65/middle_flow/block1/separable_conv1/'
        self.relu12 = nn.ReLU(inplace=True)
        self.seperate_conv10 = seperate_conv(256, 256, 3, 1, self.name_scope)
        self.name_scope = 'xception_65/middle_flow/block1/separable_conv2/'
        self.relu13 = nn.ReLU(inplace=True)
        self.seperate_conv11 = seperate_conv(256, 256, 3, 1, self.name_scope)
        self.name_scope = 'xception_65/middle_flow/block1/separable_conv3/'
        self.relu14 = nn.ReLU(inplace=True)
        self.seperate_conv12 = seperate_conv(256, 256, 3, 1, self.name_scope)
        ## TODO 逐元素相加 addition3+seperate_conv12 name:addition4

        self.name_scope = 'xception_65/middle_flow/block2/separable_conv1/'
        self.relu15 = nn.ReLU(inplace=True)
        self.seperate_conv13 = seperate_conv(256, 256, 3, 1, self.name_scope)
        self.name_scope = 'xception_65/middle_flow/block2/separable_conv2/'
        self.relu16 = nn.ReLU(inplace=True)
        self.seperate_conv14 = seperate_conv(256, 256, 3, 1, self.name_scope)
        self.name_scope = 'xception_65/middle_flow/block2/separable_conv3/'
        self.relu17 = nn.ReLU(inplace=True)
        self.seperate_conv15 = seperate_conv(256, 256, 3, 1, self.name_scope)
        ## TODO 逐元素相加 addition4+seperate_conv15  name:addition5

        self.name_scope = 'xception_65/middle_flow/block3/separable_conv1/'
        self.relu18 = nn.ReLU(inplace=True)
        self.seperate_conv16 = seperate_conv(256, 256, 3, 1, self.name_scope)
        self.name_scope = 'xception_65/middle_flow/block3/separable_conv2/'
        self.relu19 = nn.ReLU(inplace=True)
        self.seperate_conv17 = seperate_conv(256, 256, 3, 1, self.name_scope)
        self.name_scope = 'xception_65/middle_flow/block3/separable_conv3/'
        self.relu20 = nn.ReLU(inplace=True)
        self.seperate_conv18 = seperate_conv(256, 256, 3, 1, self.name_scope)
        ## TODO 逐元素相加 addition5+seperate_conv18  name:addition6

        self.name_scope = 'xception_65/middle_flow/block4/separable_conv1/'
        self.relu21 = nn.ReLU(inplace=True)
        self.seperate_conv19 = seperate_conv(256, 256, 3, 1, self.name_scope)
        self.name_scope = 'xception_65/middle_flow/block4/separable_conv2/'
        self.relu22 = nn.ReLU(inplace=True)
        self.seperate_conv20 = seperate_conv(256, 256, 3, 1, self.name_scope)
        self.name_scope = 'xception_65/middle_flow/block4/separable_conv3/'
        self.relu23 = nn.ReLU(inplace=True)
        self.seperate_conv21 = seperate_conv(256, 256, 3, 1, self.name_scope)
        ## TODO 逐元素相加 addition6+seperate_conv21  name:addition7

        self.name_scope = 'xception_65/middle_flow/block5/separable_conv1/'
        self.relu24 = nn.ReLU(inplace=True)
        self.seperate_conv22 = seperate_conv(256, 256, 3, 1, self.name_scope)
        self.name_scope = 'xception_65/middle_flow/block5/separable_conv2/'
        self.relu25 = nn.ReLU(inplace=True)
        self.seperate_conv23 = seperate_conv(256, 256, 3, 1, self.name_scope)
        self.name_scope = 'xception_65/middle_flow/block5/separable_conv3/'
        self.relu26 = nn.ReLU(inplace=True)
        self.seperate_conv24 = seperate_conv(256, 256, 3, 1, self.name_scope)
        ## TODO 逐元素相加 addition7+seperate_conv24  name:addition8

        self.name_scope = 'xception_65/middle_flow/block6/separable_conv1/'
        self.relu27 = nn.ReLU(inplace=True)
        self.seperate_conv25 = seperate_conv(256, 256, 3, 1, self.name_scope)
        self.name_scope = 'xception_65/middle_flow/block6/separable_conv2/'
        self.relu28 = nn.ReLU(inplace=True)
        self.seperate_conv26 = seperate_conv(256, 256, 3, 1, self.name_scope)
        self.name_scope = 'xception_65/middle_flow/block6/separable_conv3/'
        self.relu29 = nn.ReLU(inplace=True)
        self.seperate_conv27 = seperate_conv(256, 256, 3, 1, self.name_scope)
        ## TODO 逐元素相加 addition8+seperate_conv27  name:addition9

        self.name_scope = 'xception_65/middle_flow/block7/separable_conv1/'
        self.relu30 = nn.ReLU(inplace=True)
        self.seperate_conv28 = seperate_conv(256, 256, 3, 1, self.name_scope)
        self.name_scope = 'xception_65/middle_flow/block7/separable_conv2/'
        self.relu31 = nn.ReLU(inplace=True)
        self.seperate_conv29 = seperate_conv(256, 256, 3, 1, self.name_scope)
        self.name_scope = 'xception_65/middle_flow/block7/separable_conv3/'
        self.relu32 = nn.ReLU(inplace=True)
        self.seperate_conv30 = seperate_conv(256, 256, 3, 1, self.name_scope)
        ## TODO 逐元素相加 addition9+seperate_conv30  name:addition10

        self.name_scope = 'xception_65/middle_flow/block8/separable_conv1/'
        self.relu33 = nn.ReLU(inplace=True)
        self.seperate_conv31 = seperate_conv(256, 256, 3, 1, self.name_scope)
        self.name_scope = 'xception_65/middle_flow/block8/separable_conv2/'
        self.relu34 = nn.ReLU(inplace=True)
        self.seperate_conv32 = seperate_conv(256, 256, 3, 1, self.name_scope)
        self.name_scope = 'xception_65/middle_flow/block8/separable_conv3/'
        self.relu35 = nn.ReLU(inplace=True)
        self.seperate_conv33 = seperate_conv(256, 256, 3, 1, self.name_scope)
        ## TODO 逐元素相加 addition10+seperate_conv33  name:addition11

        self.name_scope = 'xception_65/exit_flow/block1/separable_conv1'
        self.relu36 = nn.ReLU(inplace=True)
        self.seperate_conv34 = seperate_conv(256, 256, 3, 1, self.name_scope)
        self.name_scope = 'xception_65/exit_flow/block1/separable_conv2'
        self.relu37 = nn.ReLU(inplace=True)
        self.seperate_conv35 = seperate_conv(256, 512, 3, 1, self.name_scope)
        self.name_scope = 'xception_65/exit_flow/block1/separable_conv3'
        self.relu38 = nn.ReLU(inplace=True)
        self.seperate_conv36 = seperate_conv(512, 512, 3, 1, self.name_scope)
        
        self.name_scope = 'xception_65/exit_flow/block1/shortcut/'
        ## 这里用addition11输出
        self.conv6 = conv(self.name_scope, 256, 512, kernel_size=1, stride=1)
        self.bn6 = nn.BatchNorm2d(512)
        ## TODO 逐元素相加  bn6+seperate_conv36  name:addition12

        self.name_scope = 'xception_65/exit_flow/block2/separable_conv1/'
        self.seperate_conv37 = seperate_conv(512, 512, 3, 1, self.name_scope, dilation=2, act='relu')
        self.name_scope = 'xception_65/exit_flow/block2/separable_conv2/'
        self.seperate_conv38 = seperate_conv(512, 512, 3, 1, self.name_scope, dilation=2, act='relu')
        self.name_scope = 'xception_65/exit_flow/block2/separable_conv3/'
        self.seperate_conv39 = seperate_conv(512, 768, 3, 1, self.name_scope, dilation=2, act='relu')

        self.name_scope = 'encoder/aspp0/'
        self.conv7 = conv(self.name_scope, 768, 192, 1)
        ## aspp0
        self.bn_relu1 = bn_relu(192)
        self.name_scope = 'encoder/aspp1/'
        ## aspp1
        self.seperate_conv40 = seperate_conv(768, 192, 3, 1, self.name_scope, dilation=3, act='relu')
        self.name_scope = 'encoder/aspp2/'
        ## aspp2
        self.seperate_conv41 = seperate_conv(768, 192, 3, 1, self.name_scope, dilation=6, act='relu')
        self.name_scope = 'encoder/aspp3/'
        ## aspp3
        self.seperate_conv42 = seperate_conv(768, 192, 3, 1, self.name_scope, dilation=12, act='relu')
        ## TODO concat aspp0 aspp1 aspp2 aspp3  name:concat1 chanel=192*4=768

        self.conv8 = conv(self.name_scope, 768, 192, 1)
        self.bn_relu2 = bn_relu(192)


        ## decoder部分  encode_data=bn_relu2  decode_shortcut2=block2_results[2] channel=128
        self.name_scope = 'decoder/concat/'
        self.conv9 = conv(self.name_scope, 128, 32, 1)
        self.bn_relu3 = bn_relu(32)
        self.upsampling1 = nn.UpsamplingBilinear2d(scale_factor=2)
        ## TODO concat bn_relu2 upsampling1 name:concat2 channel=192+32=224

        self.name_scope = 'decoder/separable_conv1/'
        self.seperate_conv43 = seperate_conv(224, 160, 3, 1, self.name_scope, act='relu')
        self.name_scope = 'decoder/separable_conv2/'
        self.seperate_conv44 = seperate_conv(160, 160, 3, 1, self.name_scope, act='relu')

        self.upsampling2 = nn.UpsamplingBilinear2d(scale_factor=2)

        ## decoder2部分 encode_data=upsampling2  decode_shortcut1=block1_results[1]  channel=64
        self.name_scope = 'decoder2/concat2/'
        ## 这里传入block1_results[1]
        self.conv10 = conv(self.name_scope, 64, 16, 1)
        ## 传入self.conv10
        self.bn_relu4 = bn_relu(16)
        ## 传入upsampling2
        self.upsampling3 = nn.UpsamplingBilinear2d(scale_factor=2)
        ## TODO concat  upsampling3 bn_relu4 name:concat3 channel=192+32=224

        self.name_scope = 'decoder2/separable_conv12/'
        self.seperate_conv45 = seperate_conv(224, 112, 3, 1, self.name_scope, act='relu')
        self.seperate_conv46 = seperate_conv(112, 56, 3, 1, self.name_scope, act='relu')

        self.name_scope = 'logit/'
        self.logit_conv = conv(self.name_scope, 56, label_number, 1)
        self.logit = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        relu2 = self.relu2(bn2)
        relu3 = self.relu3(relu2)
        seperate_conv1 = self.seperate_conv1(relu3)
        relu4 = self.relu4(seperate_conv1)