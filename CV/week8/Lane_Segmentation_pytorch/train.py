from tqdm import tqdm
import torch
import os
import shutil
from utils.metric import compute_iou
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.image_process import LaneDataset, ImageAug, DeformAug
from utils.image_process import ScaleAug, CutOut, ToTensor
from utils.loss import MySoftmaxCrossEntropyLoss
from model.deeplabv3plus import DeeplabV3Plus
from model.unet import ResNetUNet
from config import Config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device_list = [0]
#  train_net = 'deeplabv3p'
train_net = 'unet'
nets = {'deeplabv3p': DeeplabV3Plus, 'unet': ResNetUNet}


def train_epoch(net, epoch, dataLoader, optimizer, trainF, config):
    # 将模型设置为训练状态,此时对droupout的处理是随机失活，对bn层的处理是归一化
    # net.eval(),将模型设置为评估状态，预测的时候使用，此时对dropout的处理的
    # 使用全部神经元，并且乘以补偿系数，bn层时使用的是参数在batch下的移动平均
    net.train()
    total_mask_loss = 0.0
    dataprocess = tqdm(dataLoader)
    for batch_item in dataprocess:
        image, mask = batch_item['image'], batch_item['mask']
        if torch.cuda.is_available():
            image, mask = image.cuda(device=device_list[0]), mask.cuda(device=device_list[0])
        # optimizer.zero将每个parameter的梯度清0
        optimizer.zero_grad()
        # 输出预测的mask
        out = net(image)
        # 计算每一类的交叉熵loss
        mask_loss = MySoftmaxCrossEntropyLoss(nbclasses=config.NUM_CLASSES)(out, mask)
        # 计算总交叉熵loss
        total_mask_loss += mask_loss.item()
        # 反向传播
        mask_loss.backward()
        # 更新参数
        optimizer.step()
        dataprocess.set_description_str("epoch:{}".format(epoch))
        dataprocess.set_postfix_str("mask_loss:{:.4f}".format(mask_loss.item()))
    # 将每次的loss值写入训练的log文件中 
    trainF.write("Epoch:{}, mask loss is {:.4f} \n".format(epoch, total_mask_loss / len(dataLoader)))
    trainF.flush()


def test(net, epoch, dataLoader, testF, config):
    net.eval()
    total_mask_loss = 0.0
    dataprocess = tqdm(dataLoader)
    result = {"TP": {i:0 for i in range(8)}, "TA":{i:0 for i in range(8)}}
    for batch_item in dataprocess:
        image, mask = batch_item['image'], batch_item['mask']
        if torch.cuda.is_available():
            image, mask = image.cuda(device=device_list[0]), mask.cuda(device=device_list[0])
        out = net(image)
        mask_loss = MySoftmaxCrossEntropyLoss(nbclasses=config.NUM_CLASSES)(out, mask)
        # detach（）截断梯度的作用,截断后如果tensor中的data发生了变化，反向传播时会报错（和正向传播时候的值不一样了）
        total_mask_loss += mask_loss.detach().item()
        # dim表示对channle进行处理得到N，H，W
        pred = torch.argmax(F.softmax(out, dim=1), dim=1)
        # 计算iou
        result = compute_iou(pred, mask, result)
        dataprocess.set_description_str("epoch:{}".format(epoch))
        dataprocess.set_postfix_str("mask_loss:{:.4f}".format(mask_loss))
    testF.write("Epoch:{} \n".format(epoch))
    # 求出每一个类别的iou
    for i in range(8):
        # 计算每一类的交并比
        result_string = "{}: {:.4f} \n".format(i, result["TP"][i]/result["TA"][i])
        print(result_string)
        # 写入测试log文件
        testF.write(result_string)
    testF.write("Epoch:{}, mask loss is {:.4f} \n".format(epoch, total_mask_loss / len(dataLoader)))
    testF.flush()


def adjust_lr(optimizer, epoch):
    # Warmup 策略
    # 减缓对mini batch的提前过拟合(一开始的batch数据可能方差较大，lr过大可能使其过拟合)
    # 多训练几轮后，需要加速学习，到后面模拟学习程度比较接近local point，再减小lr值
    # 有助于保持模型深层次的稳定性
    if epoch == 0:
        lr = 1e-3
    elif epoch == 2:
        lr = 1e-2
    elif epoch == 100:
        lr = 1e-3
    elif epoch == 150:
        lr = 1e-4
    else:
        return
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # 设置model parameters
    lane_config = Config()
    if os.path.exists(lane_config.SAVE_PATH):
        shutil.rmtree(lane_config.SAVE_PATH)
    os.makedirs(lane_config.SAVE_PATH, exist_ok=True)
    trainF = open(os.path.join(lane_config.SAVE_PATH, "train.csv"), 'w')
    testF = open(os.path.join(lane_config.SAVE_PATH, "test.csv"), 'w')

    # set up dataset
    # 'pin_memory'意味着生成的Tensor数据最开始是属于内存中的索页，这样的话转到GPU的显存就会很快
    # numworkers 代表子进程数目，用来为主进程加载一个batch的数据，太大会是内存溢出
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    # 对训练集进行数据增强，对验证集不需要数据增强
    train_dataset = LaneDataset("train.csv", transform=transforms.Compose([ImageAug(), DeformAug(),
                                                                              ScaleAug(), CutOut(32, 0.5), ToTensor()]))

    train_data_batch = DataLoader(train_dataset, batch_size=len(device_list), shuffle=True, drop_last=True, **kwargs)
    val_dataset = LaneDataset("val.csv", transform=transforms.Compose([ToTensor()]))

    val_data_batch = DataLoader(val_dataset, batch_size=len(device_list), shuffle=False, drop_last=False, **kwargs)

    # build model
    net = nets[train_net](lane_config)
    if torch.cuda.is_available():
        net = net.cuda(device=device_list[0])
        net = torch.nn.DataParallel(net, device_ids=device_list)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lane_config.BASE_LR,
    #                             momentum=0.9, weight_decay=lane_config.WEIGHT_DECAY)
    optimizer = torch.optim.Adam(net.parameters(), lr=lane_config.BASE_LR, weight_decay=lane_config.WEIGHT_DECAY)

    # Training and test
    for epoch in range(lane_config.EPOCHS):
        # adjust_lr(optimizer, epoch)
        train_epoch(net, epoch, train_data_batch, optimizer, trainF, lane_config)
        test(net, epoch, val_data_batch, testF, lane_config)
        # net.module.state_dict()
        if epoch % 2 == 0:
            torch.save({'state_dict': net.state_dict()}, os.path.join(os.getcwd(), lane_config.SAVE_PATH, "laneNet{}.pth.tar".format(epoch)))
    trainF.close()
    testF.close()
    torch.save({'state_dict': net.state_dict()}, os.path.join(os.getcwd(), lane_config.SAVE_PATH, "finalNet.pth.tar"))

if __name__ == "__main__":
    main()
