import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from models.unet_party import conv_bn_layer,conv_layer

class unet_simple(nn.Module):
    def __init__(self, label_number, img_size):
        super().__init__()
        encoder_depth = [3, 4, 5, 3]
        encoder_filters = [64, 128, 256, 512]
        decoder_depth = [2, 3, 3, 2]
        decoder_filters = [256, 128, 64, 32]

        self.pre_conv1 = conv_bn_layer(3, 32, kernel_size=3, stride=2, act='relu')
        self.pre_conv2 = conv_bn_layer(32, 32, kernel_size=3, act='relu')
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        ## encoder block=0
        block = 0
        self.encoder_block0_conv1 = conv_bn_layer(32, encoder_filters[block], kernel_size=3, act='relu')
        self.encoder_block0_conv2 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3)
        self.encoder_block0_convert1 = conv_bn_layer(32, encoder_filters[block], kernel_size=1)
        ## TODO 逐元素相加
        self.encoder_block0_conv3 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3, act='relu')
        self.encoder_block0_conv4 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加
        self.encoder_block0_conv5 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3, act='relu')
        self.encoder_block0_conv6 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加  conv0

        ## encoder block=1
        block = 1
        self.encoder_block1_conv1 = conv_bn_layer(64, encoder_filters[block], kernel_size=3, act='relu')
        self.encoder_block1_conv2 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3, stride=2)
        self.encoder_block1_convert = conv_bn_layer(64, encoder_filters[block], kernel_size=3, stride=2)
        ## TODO 逐元素相加
        self.encoder_block1_conv3 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3, act='relu')
        self.encoder_block1_conv4 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加
        self.encoder_block1_conv5 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3, act='relu')
        self.encoder_block1_conv6 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加
        self.encoder_block1_conv7 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3, act='relu')
        self.encoder_block1_conv8 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加  conv1

        ## encoder block=2
        block = 2
        self.encoder_block2_conv1 = conv_bn_layer(128, encoder_filters[block], kernel_size=3, act='relu')
        self.encoder_block2_conv2 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3, stride=2)
        self.encoder_block2_convert1 = conv_bn_layer(128, encoder_filters[block], kernel_size=1, stride=2)
        ## TODO 逐元素相加
        self.encoder_block2_conv3 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3, act='relu')
        self.encoder_block2_conv4 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加
        self.encoder_block2_conv5 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3, act='relu')
        self.encoder_block2_conv6 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加
        self.encoder_block2_conv7 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3, act='relu')
        self.encoder_block2_conv8 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加
        self.encoder_block2_conv9 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3, act='relu')
        self.encoder_block2_conv10 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加  conv2

        ## encoder block=3
        block=3
        self.encoder_block3_conv1 = conv_bn_layer(256, encoder_filters[block], kernel_size=3, act='relu')
        self.encoder_block3_conv2 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3, stride=2)
        self.encoder_block3_convert1 = conv_bn_layer(256, encoder_filters[block], kernel_size=1, stride=2)
        ## TODO 逐元素相加
        self.encoder_block3_conv3 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3, act='relu')
        self.encoder_block3_conv4 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加
        self.encoder_block3_conv5 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3, act='relu')
        self.encoder_block3_conv6 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加  conv3

        ## decoder block=0
        block=0
        self.decoder_block0_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.decoder_block0_conv1 = conv_bn_layer(512, decoder_filters[block], kernel_size=3, act='relu')
        self.decoder_block0_conv2 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        self.decoder_block0_convert = conv_bn_layer(512, decoder_filters[block], kernel_size=1)
        ## TODO 逐元素相加
        self.decoder_block0_concat_conv = conv_bn_layer(256, 128, kernel_size=1, act='relu')
        ## TODO 拼接把decoder_block0_convert的结果和decoder_block0_concat_conv的结果相加 shape=128+256=384
        self.decoder_block0_conv3 = conv_bn_layer(384, decoder_filters[block], kernel_size=3, act='relu')
        self.decoder_block0_conv4 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        self.decoder_block0_convert1 = conv_bn_layer(384, decoder_filters[block], kernel_size=1)
        ## TODO 逐元素相加
        self.decoder_block0_conv5 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3, act='relu')
        self.decoder_block0_conv6 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加

        ## decoder block=1
        block = 1
        self.decoder_block1_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.decoder_block1_conv1 = conv_bn_layer(256, decoder_filters[block], kernel_size=3, act='relu')
        self.decoder_block1_conv2 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        self.decoder_block1_convert = conv_bn_layer(256, decoder_filters[block], kernel_size=1)
        ## TODO 逐元素相加
        self.decoder_block1_concat_conv = conv_bn_layer(128, 64, kernel_size=1, act='relu')
        ## TODO 拼接把decoder_block1_convert的结果和decoder_block1_concat_conv的结果相加 shape=64+128=192
        self.decoder_block1_conv3 = conv_bn_layer(192, decoder_filters[block], kernel_size=3, act='relu')
        self.decoder_block1_conv4 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        self.decoder_block1_convert1 = conv_bn_layer(192, decoder_filters[block], kernel_size=1)
        ## TODO 逐元素相加
        self.decoder_block1_conv5 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3, act='relu')
        self.decoder_block1_conv6 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加
        self.decoder_block1_conv7 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3, act='relu')
        self.decoder_block1_conv8 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加


        ## decoder block=2
        block = 2
        self.decoder_block2_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.decoder_block2_conv1 = conv_bn_layer(128, decoder_filters[block], kernel_size=3, act='relu')
        self.decoder_block2_conv2 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        self.decoder_block2_convert = conv_bn_layer(128, decoder_filters[block], kernel_size=1)
        ## TODO 逐元素相加
        self.decoder_block2_concat_conv = conv_bn_layer(64, 32, kernel_size=1, act='relu')
        ## TODO 拼接把decoder_block2_convert的结果和decoder_block2_concat_conv的结果相加 shape=32+64=96
        self.decoder_block2_conv3 = conv_bn_layer(96, decoder_filters[block], kernel_size=3, act='relu')
        self.decoder_block2_conv4 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        self.decoder_block2_convert1 = conv_bn_layer(96, decoder_filters[block], kernel_size=1)
        ## TODO 逐元素相加
        self.decoder_block2_conv5 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3, act='relu')
        self.decoder_block2_conv6 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加
        self.decoder_block2_conv7 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3, act='relu')
        self.decoder_block2_conv8 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加

        ## decoder block=3
        block = 3
        self.decoder_block3_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.decoder_block3_conv1 = conv_bn_layer(64, decoder_filters[block], kernel_size=3, act='relu')
        self.decoder_block3_conv2 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        self.decoder_block3_convert = conv_bn_layer(64, decoder_filters[block], kernel_size=1)
        ## TODO 逐元素相加
        self.decoder_block3_concat_conv = conv_bn_layer(32, 16, kernel_size=1, act='relu')
        ## TODO 拼接把decoder_block3_convert的结果和decoder_block3_concat_conv的结果相加 shape=32+16=48
        self.decoder_block3_conv3 = conv_bn_layer(48, decoder_filters[block], kernel_size=3, act='relu')
        self.decoder_block3_conv4 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        self.decoder_block3_convert1 = conv_bn_layer(48, decoder_filters[block], kernel_size=1)
        ## TODO 逐元素相加
        self.decoder_block3_conv5 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3, act='relu')
        self.decoder_block3_conv6 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加

        self.decoder_block4_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.decoder_block4_conv1 = conv_bn_layer(32, 32, kernel_size=3, act='relu')
        self.decoder_block4_conv2 = conv_bn_layer(32, 32, kernel_size=3)
        ## TODO 逐元素相加
        self.decoder_block4_conv3 = conv_bn_layer(32, 16, kernel_size=3, act='relu')
        self.decoder_block4_conv4 = conv_bn_layer(16, 16, kernel_size=3)
        self.decoder_block4_convert = conv_bn_layer(32, 16, kernel_size=1)
        
        ## label_number
        self.logit = conv_layer(16, label_number, kernel_size=1, act=None)
        
    def forward(self, inputs):
        pre_conv1 = self.pre_conv1(inputs)
        pre_conv2 = self.pre_conv2(pre_conv1)
        max_pool1 = self.max_pool1(pre_conv2)

        encoder_block0_conv1 = self.encoder_block0_conv1(max_pool1)
        encoder_block0_conv2 = self.encoder_block0_conv2(encoder_block0_conv1)
        encoder_block0_convert1 = self.encoder_block0_convert1(max_pool1)
        encoder_block0_convert1 = encoder_block0_convert1 + encoder_block0_conv2
        
        encoder_block0_conv3 = self.encoder_block0_conv3(encoder_block0_convert1)
        encoder_block0_conv4 = self.encoder_block0_conv4(encoder_block0_conv3)
        encoder_block0_conv4 = encoder_block0_conv4 + encoder_block0_convert1

        encoder_block0_conv5 = self.encoder_block0_conv5(encoder_block0_conv4)
        encoder_block0_conv6 = self.encoder_block0_conv6(encoder_block0_conv5)
        encoder_block0_conv6 = encoder_block0_conv6 + encoder_block0_conv4

        encoder_block1_conv1 = self.encoder_block1_conv1(encoder_block0_conv6)
        encoder_block1_conv2 = self.encoder_block1_conv2(encoder_block1_conv1)
        encoder_block1_convert = self.encoder_block1_convert(encoder_block0_conv6)
        encoder_block1_convert = encoder_block1_convert + encoder_block1_conv2

        encoder_block1_conv3 = self.encoder_block1_conv3(encoder_block1_convert)
        encoder_block1_conv4 = self.encoder_block1_conv4(encoder_block1_conv3)
        encoder_block1_conv4 = encoder_block1_conv4 + encoder_block1_convert

        encoder_block1_conv5 = self.encoder_block1_conv5(encoder_block1_conv4)
        encoder_block1_conv6 = self.encoder_block1_conv6(encoder_block1_conv5)
        encoder_block1_conv6 = encoder_block1_conv6 + encoder_block1_conv4

        encoder_block1_conv7 = self.encoder_block1_conv7(encoder_block1_conv6)
        encoder_block1_conv8 = self.encoder_block1_conv8(encoder_block1_conv7)
        encoder_block1_conv8 = encoder_block1_conv8 + encoder_block1_conv6

        encoder_block2_conv1 = self.encoder_block2_conv1(encoder_block1_conv8)
        encoder_block2_conv2 = self.encoder_block2_conv2(encoder_block2_conv1)
        encoder_block2_convert1 = self.encoder_block2_convert1(encoder_block1_conv8)
        encoder_block2_convert1 = encoder_block2_convert1 + encoder_block2_conv2

        encoder_block2_conv3 = self.encoder_block2_conv3(encoder_block2_convert1)
        encoder_block2_conv4 = self.encoder_block2_conv4(encoder_block2_conv3)
        encoder_block2_conv4 = encoder_block2_conv4 + encoder_block2_convert1

        encoder_block2_conv5 = self.encoder_block2_conv5(encoder_block2_conv4)
        encoder_block2_conv6 = self.encoder_block2_conv6(encoder_block2_conv5)
        encoder_block2_conv6 = encoder_block2_conv6 + encoder_block2_conv4

        encoder_block2_conv7 = self.encoder_block2_conv7(encoder_block2_conv6)
        encoder_block2_conv8 = self.encoder_block2_conv8(encoder_block2_conv7)
        encoder_block2_conv8 = encoder_block2_conv8 + encoder_block2_conv6

        encoder_block2_conv9 = self.encoder_block2_conv9(encoder_block2_conv8)
        encoder_block2_conv10 = self.encoder_block2_conv10(encoder_block2_conv9)
        encoder_block2_conv10 = encoder_block2_conv10 + encoder_block2_conv8

        encoder_block3_conv1 = self.encoder_block3_conv1(encoder_block2_conv10)
        encoder_block3_conv2 = self.encoder_block3_conv2(encoder_block3_conv1)
        encoder_block3_convert1 = self.encoder_block3_convert1(encoder_block2_conv10)
        encoder_block3_convert1 = encoder_block3_convert1 + encoder_block3_conv2

        encoder_block3_conv3 = self.encoder_block3_conv3(encoder_block3_convert1)
        encoder_block3_conv4 = self.encoder_block3_conv4(encoder_block3_conv3)
        encoder_block3_conv4 = encoder_block3_conv4 + encoder_block3_convert1

        encoder_block3_conv5 = self.encoder_block3_conv5(encoder_block3_conv4)
        encoder_block3_conv6 = self.encoder_block3_conv6(encoder_block3_conv5)
        encoder_block3_conv6 = encoder_block3_conv6 + encoder_block3_conv4

        decoder_block0_upsampling = self.decoder_block0_upsampling(encoder_block3_conv6)
        decoder_block0_conv1 = self.decoder_block0_conv1(decoder_block0_upsampling)
        decoder_block0_conv2 = self.decoder_block0_conv2(decoder_block0_conv1)
        decoder_block0_convert = self.decoder_block0_convert(decoder_block0_upsampling)
        decoder_block0_convert = decoder_block0_convert + decoder_block0_conv2

        ## 拼接把decoder_block0_convert的结果和decoder_block0_concat_conv的结果相加 shape=128+256=384
        decoder_block0_concat_conv = self.decoder_block0_concat_conv(encoder_block2_conv10)
        decoder_block0_cat = torch.cat((decoder_block0_concat_conv, decoder_block0_convert), 1)

        decoder_block0_conv3 = self.decoder_block0_conv3(decoder_block0_cat)
        decoder_block0_conv4 = self.decoder_block0_conv4(decoder_block0_conv3)
        decoder_block0_convert1 = self.decoder_block0_convert1(decoder_block0_cat)
        decoder_block0_convert1 = decoder_block0_convert1 + decoder_block0_conv4

        decoder_block0_conv5 = self.decoder_block0_conv5(decoder_block0_convert1)
        decoder_block0_conv6 = self.decoder_block0_conv6(decoder_block0_conv5)
        decoder_block0_conv6 = decoder_block0_conv6 + decoder_block0_convert1

        decoder_block1_upsampling = self.decoder_block1_upsampling(decoder_block0_conv6)
        decoder_block1_conv1 = self.decoder_block1_conv1(decoder_block1_upsampling)
        decoder_block1_conv2 = self.decoder_block1_conv2(decoder_block1_conv1)
        decoder_block1_convert = self.decoder_block1_convert(decoder_block1_upsampling)
        decoder_block1_convert = decoder_block1_convert + decoder_block1_conv2

        ## 拼接把decoder_block1_convert的结果和decoder_block1_concat_conv的结果相加 shape=64+128=192
        decoder_block1_concat_conv = self.decoder_block1_concat_conv(encoder_block1_conv8)
        decoder_block1_cat = torch.cat((decoder_block1_concat_conv, decoder_block1_convert), 1)
        decoder_block2_conv3 = self.decoder_block2_conv3(decoder_block1_cat)
        decoder_block2_conv4 = self.decoder_block2_conv4(decoder_block2_conv3)
        decoder_block2_convert1 = self.decoder_block2_convert1(decoder_block1_cat)
        decoder_block2_convert1 = decoder_block2_convert1 + decoder_block2_conv4

        decoder_block2_conv5 = self.decoder_block2_conv5(decoder_block2_convert1)
        decoder_block2_conv6 = self.decoder_block2_conv6(decoder_block2_conv5)
        decoder_block2_conv6 = decoder_block2_conv6 + decoder_block2_convert1

        decoder_block2_conv7 = self.decoder_block2_conv7(decoder_block2_conv6)
        decoder_block2_conv8 = self.decoder_block2_conv8(decoder_block2_conv7)
        decoder_block2_conv8 = decoder_block2_conv8 + decoder_block2_conv6

        decoder_block3_upsampling = self.decoder_block3_upsampling(decoder_block2_conv8)
        decoder_block3_conv1 = self.decoder_block3_conv1(decoder_block3_upsampling)
        decoder_block3_conv2 = self.decoder_block3_conv2(decoder_block3_conv1)
        decoder_block3_convert = self.decoder_block3_convert(decoder_block3_upsampling)
        decoder_block3_convert = decoder_block3_convert + decoder_block3_conv2

        ## 拼接把decoder_block3_convert的结果和decoder_block3_concat_conv的结果相加 shape=32+16=48
        decoder_block3_concat_conv = self.decoder_block3_concat_conv(pre_conv2)
        decoder_block3_conv3 = self.decoder_block3_conv3(decoder_block3_concat_conv)
        decoder_block3_conv4 = self.decoder_block3_conv4(decoder_block3_conv3)
        decoder_block3_convert1 = self.decoder_block3_convert1(decoder_block3_concat_conv)
        decoder_block3_convert1 = decoder_block3_convert1 + decoder_block3_conv4

        decoder_block3_conv5 = self.decoder_block3_conv5(decoder_block3_convert1)
        decoder_block3_conv6 = self.decoder_block3_conv6(decoder_block3_conv5)
        decoder_block3_conv6 = decoder_block3_conv6 + decoder_block3_convert1

        decoder_block4_upsampling = self.decoder_block4_upsampling(decoder_block3_conv6)
        decoder_block4_conv1 = self.decoder_block4_conv1(decoder_block4_upsampling)
        decoder_block4_conv2 = self.decoder_block4_conv2(decoder_block4_conv1)
        decoder_block4_conv2 = decoder_block4_conv2 + decoder_block4_upsampling

        decoder_block4_conv3 = self.decoder_block4_conv3(decoder_block4_conv2)
        decoder_block4_conv4 = self.decoder_block4_conv4(decoder_block4_conv3)
        decoder_block4_convert = self.decoder_block4_convert(decoder_block4_conv2)
        decoder_block4_convert = decoder_block4_convert + decoder_block4_conv4

        logit = self.logit(decoder_block4_convert)

        return logit