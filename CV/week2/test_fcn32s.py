import torch 
import torch.nn as nn
# 测试代码 fcn32s
n_class=21
self_conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
self_relu1_1 = nn.ReLU(inplace=True)
self_conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
self_relu1_2 = nn.ReLU(inplace=True)
self_pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2
self_conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
self_relu2_1 = nn.ReLU(inplace=True)
self_conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
self_relu2_2 = nn.ReLU(inplace=True)
self_pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

# conv3
self_conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
self_relu3_1 = nn.ReLU(inplace=True)
self_conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
self_relu3_2 = nn.ReLU(inplace=True)
self_conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
self_relu3_3 = nn.ReLU(inplace=True)
self_pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

# conv4
self_conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
self_relu4_1 = nn.ReLU(inplace=True)
self_conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
self_relu4_2 = nn.ReLU(inplace=True)
self_conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
self_relu4_3 = nn.ReLU(inplace=True)
self_pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

# conv5
self_conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
self_relu5_1 = nn.ReLU(inplace=True)
self_conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
self_relu5_2 = nn.ReLU(inplace=True)
self_conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
self_relu5_3 = nn.ReLU(inplace=True)
self_pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

# fc6
self_fc6 = nn.Conv2d(512, 4096, 7)
self_relu6 = nn.ReLU(inplace=True)
self_drop6 = nn.Dropout2d()

# fc7
self_fc7 = nn.Conv2d(4096, 4096, 1)
self_relu7 = nn.ReLU(inplace=True)
self_drop7 = nn.Dropout2d()

self_score_fr = nn.Conv2d(4096, n_class, 1)
self_upscore = nn.ConvTranspose2d(n_class, n_class, 64, stride=32,
                                  bias=False)
def run(x):
   h = x
   print(0, h.shape)
   h = self_relu1_1(self_conv1_1(h))
   print(0, h.shape)
   h = self_relu1_2(self_conv1_2(h))
   print(0, h.shape)
   h = self_pool1(h)

   print(1, h.shape)
   h = self_relu2_1(self_conv2_1(h))
   h = self_relu2_2(self_conv2_2(h))
   h = self_pool2(h)

   print(2, h.shape)
   h = self_relu3_1(self_conv3_1(h))
   h = self_relu3_2(self_conv3_2(h))
   h = self_relu3_3(self_conv3_3(h))
   h = self_pool3(h)

   print(3, h.shape)
   h = self_relu4_1(self_conv4_1(h))
   h = self_relu4_2(self_conv4_2(h))
   h = self_relu4_3(self_conv4_3(h))
   h = self_pool4(h)

   print(4, h.shape)
   h = self_relu5_1(self_conv5_1(h))
   h = self_relu5_2(self_conv5_2(h))
   h = self_relu5_3(self_conv5_3(h))
   h = self_pool5(h)

   print(5, h.shape)
   h = self_relu6(self_fc6(h))
   h = self_drop6(h)

   print(6, h.shape)
   h = self_relu7(self_fc7(h))
   h = self_drop7(h)

   print(7, h.shape)
   h = self_score_fr(h)
   print(8, h.shape)

   h = self_upscore(h)
   print(9, h.shape)
   h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()

   print(10, h.shape)
   return h

x = torch.randn(1, 3, 224, 224)
if __name__ == "__main__":
    run(x)
# 分析
#卷积：
#fout = (fin + 2*padding - kernel)/stride + 1
#转置卷积：
#fin = (fout - 1) * stride + kernel - 2 * padding 
#padding100的作用：
#如果padding=1，kernel为3，则图片尺寸不发生变化
#经过5次maxpool后，图片尺寸变为n/32
#经过一个7*7卷积,padding=0, 图片尺寸变为 (n/32 -7)/1 + 1 = (n - 192)/32
#对于图片尺寸小于192，在7*7卷积时会报错。
#padding100 后，第一次卷积，图片尺寸变为 （n + 200 -3）+1 = n + 198
#经过5次maxpool后，图片尺寸变为（n+198)/32
#经过7*7 卷积后，图片尺寸变为:  ((n+198)/32 -7)/1 + 1 = (n + 6)/32

#vgg32s:
#经过转置卷积上采样32倍，stride=32,kernel=64, ((n+6)/32 -1)*32 + 64 = n + 38
#所以进行裁剪时，裁剪值为19

#vgg16s:
# 先2倍上采样
# 假设第4次pool后尺寸为m，channel 数目为512，则第五次后变为m/2，7*7 卷积后变为(m/2 -7)/1 +1 = n/2 - 6,
# 经过一个上采样2倍，(m/2 -6 -1)*2 + 4 = m - 10,比原图像小了10，所以将原图像裁剪值为5
# 再16倍上采样
# 四次pool后，加上裁剪，图片尺寸和原图像关系为： (n+198)/16 -10 = (n +38)/16
# 进行上采样16  ((n+38)/16 -1)*16 +32 = n+54，所以最后还需要裁剪27

#vgg8s:
#先进行2倍上采样，同16s，裁剪值为5
#再进行一个2倍上采样 k/4 - 7 +1 =  k/4 - 6  经过第一个上采样： (k/4 -6 -1)*2+ 4 = k/2 -10
# 再经过一个2倍 (k/2 -10 -1)*2 +4 = k -18 ，所以裁剪值为9
#再经过一个8倍上采样 (n+198)/8 -18 = (n +54)/8    ((n+54)/8 -1)*8 +16 =n+62
#所以最后一个裁剪为31

#以下是输入为1*1 时各层网络的输出尺寸： 
#[Running] python - STDIN
#
#--------------------
#0 torch.Size([1, 3, 1, 1])
#0 torch.Size([1, 64, 199, 199])
#0 torch.Size([1, 64, 199, 199])
#1 torch.Size([1, 64, 100, 100])
#2 torch.Size([1, 128, 50, 50])
#3 torch.Size([1, 256, 25, 25])
#4 torch.Size([1, 512, 13, 13])
#5 torch.Size([1, 512, 7, 7])
#6 torch.Size([1, 4096, 1, 1])
#7 torch.Size([1, 4096, 1, 1])
#8 torch.Size([1, 21, 1, 1])
#9 torch.Size([1, 21, 64, 64])
#10 torch.Size([1, 21, 1, 1])

#[Done] exited with code=0 in 6.903468 seconds
#以下是n为224是各网络层的输出
#[Running] python - STDIN
#
#--------------------
#0 torch.Size([1, 3, 224, 224])
#0 torch.Size([1, 64, 422, 422])
#0 torch.Size([1, 64, 422, 422])
#1 torch.Size([1, 64, 211, 211])
#2 torch.Size([1, 128, 106, 106])
#3 torch.Size([1, 256, 53, 53])
#4 torch.Size([1, 512, 27, 27])
#5 torch.Size([1, 512, 14, 14])
#下面应该是7，为啥是向上取整？？
#6 torch.Size([1, 4096, 8, 8])
#7 torch.Size([1, 4096, 8, 8])
#8 torch.Size([1, 21, 8, 8])
# n 为224 这里n + 64？？上面没有向下取整导致的。
#9 torch.Size([1, 21, 288, 288])
#10 torch.Size([1, 21, 224, 224])
#
#[Done] exited with code=0 in 9.094105 seconds
