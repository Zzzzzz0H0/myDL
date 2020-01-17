import torch
import torch.nn as nn
class FCN8s(nn.Module):
    def __init__(self, n_class=21):
        super(FCN8s, self).__init__()
        #layer1
        self.cov11 = nn.Conv2d(3, 64, 3, padding=100)
        self.cov12 = nn.Conv2d(64, 64, 3, padding=1)
        self.maxp1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        #layer2
        self.cov21 = nn.Conv2d(64, 128, 3, padding=1)
        self.cov22 = nn.Conv2d(128, 128, 3, padding=1)
        self.maxp2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        #layer3
        self.cov31 = nn.Conv2d(128, 256, 3, padding=1)
        self.cov32 = nn.Conv2d(256, 256, 3, padding=1)
        self.cov33 = nn.Conv2d(256, 256, 3, padding=1)
        self.maxp3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        #layer4
        self.cov41 = nn.Conv2d(256, 512, 3, padding=1)
        self.cov42= nn.Conv2d(512, 512, 3, padding=1)
        self.cov43 = nn.Conv2d(512, 512, 3, padding=1)
        self.maxp4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        #layer5 
        #参数太多，第5层通道未翻倍
        self.cov51 = nn.Conv2d(512, 512, 3, padding=1)
        self.cov52 = nn.Conv2d(512, 512, 3, padding=1)
        self.cov53 = nn.Conv2d(512, 512, 3, padding=1)
        self.maxp5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        # layer6
        self.cov6 = nn.Conv2d(512, 4096, 7)
        # layer7
        self.cov71 = nn.Conv2d(4096, 4096, 1)
        # 全链接层 
        self.cov72 = nn.Conv2d(4096, n_class, 1)
        # 转置卷积
        self.upsample2 = nn.ConvTranspose2d(n_class, n_class, kernel_size=4, stride=2)
        self.upsample4 = nn.ConvTranspose2d(n_class, n_class, kernel_size=4, stride=2)
        self.upsample8 = nn.ConvTranspose2d(n_class, n_class, kernel_size=16, stride=8)
        # 1*1 卷积用来调整上采样时的channel数，out_channel都调整为n_class,因为输出需要n_class
        self.score1 = nn.Conv2d(512, n_class, kernel_size=1)
        self.score2 = nn.Conv2d(256, n_class, kernel_size=1)
        #relu 和dropout
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()

    def forward(self, x):
        print('forward function')
        # layer1
        print('before layer1:', x.shape)
        h = self.relu(self.cov11(x))
        h = self.relu(self.cov12(h))
        h = self.maxp1(h)
        print('after layer1', h.shape)
        # layer2
        h = self.relu(self.cov21(h))
        h = self.relu(self.cov22(h))
        h = self.maxp2(h)
        print('after layer2:', h.shape)
        # layer3
        h = self.relu(self.cov31(h))
        h = self.relu(self.cov32(h))
        h = self.relu(self.cov33(h))
        h = self.maxp3(h)
        pool3 = h # 保存layer3后结果
        print('after layer3:', h.shape)
        # layer4
        h = self.relu(self.cov41(h))
        h = self.relu(self.cov42(h))
        h = self.relu(self.cov43(h))
        h = self.maxp4(h)
        pool4 = h # 保存layer4 后的结果
        print('after layer4:', h.shape)
        # layer5
        h = self.relu(self.cov51(h))
        h = self.relu(self.cov52(h))
        h = self.relu(self.cov53(h))
        h = self.maxp5(h)
        print('after layer5:', h.shape)
        # layer6
        h = self.cov6(h)
        h = self.dropout(h)
        print('after layer6:', h.shape)
        # layer7
        h = self.cov71(h)
        h = self.dropout(h)
        # 全链接层
        h = self.cov72(h)
        print('after layer7:', h.shape)

        # 2倍上采样, 裁剪值为5
        h = self.upsample2(h)
        h_score1  = self.score1(pool4)
        h_score1 = h_score1[:, :, 5:5+h.size()[2],5:5+h.size()[3]]
        h = h + h_score1
        print('after upsample2:', h.shape)

        # 4倍上采样, 裁剪值为9
        h = self.upsample4(h)
        h_score2  = self.score2(pool3)
        h_score2 = h_score2[:, :, 9:9+h.size()[2], 9:9+h.size()[3]]
        h = h + h_score2
        print('after upsample4:', h.shape)

        # 32倍上采样, 裁剪值为31
        h = self.upsample8(h)
        #h_score2  = self.score2(pool3)
        #h_score2 = h_score2[:, :, 9:9+h.size()[2], 9:9+h.size()[3]]
        print(h.shape)
        h = h[:, :, 31:31+x.size()[2], 31:31+x.size()[3]]
        print('after upsample32:', h.shape)

if __name__ == "__main__":
    module = FCN8s()
    module.forward(x=torch.randn(1, 3, 224, 224))
# output:
#[Running] python - STDIN
#--------------------
#forward function
#before layer1: torch.Size([1, 3, 224, 224])
#after layer1 torch.Size([1, 64, 211, 211])
#after layer2: torch.Size([1, 128, 106, 106])
#after layer3: torch.Size([1, 256, 53, 53])
#after layer4: torch.Size([1, 512, 27, 27])
#after layer5: torch.Size([1, 512, 14, 14])
# 同32s，这里应该是7，没有向下取整，导致数值和退到结果不一致
#after layer6: torch.Size([1, 4096, 8, 8])
#after layer7: torch.Size([1, 21, 8, 8])
#after upsample2: torch.Size([1, 21, 18, 18])
#after upsample4: torch.Size([1, 21, 38, 38])
#torch.Size([1, 21, 312, 312])
#after upsample32: torch.Size([1, 21, 224, 224])
#
#[Done] exited with code=0 in 8.323093 seconds
