import torch 
import torch.nn as nn 
# 2代表样本数量，3代表每个类别的分数
#  input = torch.randn(2, 3)
input = torch.tensor([[-0.3580,  0.2772, -0.7072],
                      [-0.0766,  0.6970,  0.4561]])
target = torch.tensor([1, 2])
print('input', input)
# 先进行softmax处理，范围（0，1）
# dim为1 是因为一行的三列得分和要为1
sm = nn.Softmax(dim=1)
input = sm(input)
print('after softmax', input)
#  tensor([[0.1364, 0.6266, 0.2370],
        #  [0.1575, 0.4315, 0.4110]])
# 再去自然对数，范围（-∞，0）
input = torch.log(input)
print('after log', input)
#  after log tensor([[-1.2789, -0.6437, -1.6281],
        #  [-1.5833, -0.8097, -1.0506]])
input = torch.tensor([[-0.3580,  0.2772, -0.7072],
                      [-0.0766,  0.6970,  0.4561]])
# LogSoftmax 等于先进行Softmax再进行log
soft_log = nn.LogSoftmax(dim=1)
print('after softmax log', soft_log(input))
#  after softmax log tensor([[-1.2789, -0.6437, -1.6281],
       #  [-1.5833, -0.8097, -1.0506]])
nll = nn.NLLLoss()
nll_loss = nll(input, target)
print('nll_loss', nll_loss)
# nll_loss tensor(0.8472)
print((0.6437 + 1.0506)/2)
#  0.8471500000000001
# 交叉熵就是SUM(-plog(q)) p为期望概率，q为实际概率
# 第一个样本预测为1，用onehot表示为[0,1,0]乘以-0.6437再取反即为0.0637
# 同样计算样本2的值，然后取均值即为交叉熵
# nll_loss做的事就是将2转为onehot然后乘以实际概率值（其实就是取出第二个值），再求和取均值
input = torch.tensor([[-0.3580,  0.2772, -0.7072],
                      [-0.0766,  0.6970,  0.4561]])
cross = nn.CrossEntropyLoss()
# cross loss就是先softmax，再log,再求nll_loss 
cross_loss = cross(input, target)
print('cross_loss', cross_loss)
#  cross_loss tensor(0.8472)
