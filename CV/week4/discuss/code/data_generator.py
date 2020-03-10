from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.image_process import LaneDataset, ImageAug, DeformAug
from utils.image_process import ScaleAug, CutOut, ToTensor


kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
training_dataset = LaneDataset("train.csv", transform=transforms.Compose([ImageAug(), DeformAug(),
                                                                          ScaleAug(), CutOut(32,0.5), ToTensor()]))


#真正开始处理数据
training_data_batch = DataLoader(training_dataset, batch_size=16,
                                 shuffle=True, drop_last=True, **kwargs)
"""
102
20
2"""
for batch_item in training_data_batch:
    image, mask = batch_item['image'], batch_item['mask'] #得到的就是经过数据处理的
    if torch.cuda.is_available():
        image, mask = image.cuda(), mask.cuda()

    #如果有模型的话，就是讲数据加载进模型开始训练了
    #  prediction = model(image)
    # loss = f (prediction,mask)

    print(image.size())
    print(mask.size())





