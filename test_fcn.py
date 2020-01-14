import torch 
import torch.nn as nn
n_class=21
self_conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
self_relu1_1 = nn.ReLU(inplace=True)
self_conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
self_relu1_2 = nn.ReLU(inplace=True)
self_pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

# conv2
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
   h = ?!?jedi=0, self_relu1_1(self_conv1_1(h))?!? (*_*value*_*, ..., sep, end, file, flush) ?!?jedi?!?
   print(01, h.shape)
   h = self_relu1_2(self_conv1_2(h))
   print(02, h.shape)
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

x = torch.randn(1, 3, 1, 1)
if __name__ == "__main__":
    run(x)
