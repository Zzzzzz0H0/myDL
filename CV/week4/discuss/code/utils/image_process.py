import os
import cv2
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from imgaug import augmentere as iaa
from utils.process_labels import encode_labels, decode_labels, decode_color_labels


sometimes = lambda aug: iaa.Sometimes(0.5, aug)


# crop the image to discard useless parts
def crop_resize_data(image, label=None, image_size=(1024, 384), offset=690):
    """
    Attention:
    h,w, c = image.shape
    cv2.resize(image,(w,h))
    """
    roi_image = image[offset:, :]
    if label is not None:
        roi_label = label[offset:, :]
        """
        55557777
        5555557777
        5555667777 在边界出现了一些不应该出现的类别或者错误"""
        train_image = cv2.resize(roi_image, image_size, interpolation=cv2.INTER_LINEAR) #image做的是线性插值
        train_label = cv2.resize(roi_label, image_size, interpolation=cv2.INTER_NEAREST) #label做的是最邻近插值
        return train_image, train_label
    else:
        train_image = cv2.resize(roi_image, image_size, interpolation=cv2.INTER_LINEAR)
        return train_image


class LaneDataset(Dataset): #以dataset为父类，声明一个子类

    #__init__ 将所有的数据都加载进来
    def __init__(self, csv_file, transform=None):
        super(LaneDataset, self).__init__()
        self.data = pd.read_csv(os.path.join(os.getcwd(), "data_list", csv_file), header=None,
                                  names=["image",
                                         "label"])

        #将图像的地址加载进来，需要保证一条真实的数据对应一个真实的index
        #这两列对应的是真实值的地址
        self.images = self.data["image"].values
        self.labels = self.data["label"].values

        self.transform = transform

    #数据集的大小，在数据生成过程中，可以帮助设置batch_size的大小，和epoch
    def __len__(self):
        return self.labels.shape[0] #总长度

    #如何得到最终网络输进去的数据，包括图像预处理，数据增强后的图像
    def __getitem__(self, idx):

        # idx是数据生成过程中给出的值
        ori_image = cv2.imread(self.images[idx])  #self.image 存储的是地址
        ori_mask = cv2.imread(self.labels[idx], cv2.IMREAD_GRAYSCALE)


        train_img, train_mask = crop_resize_data(ori_image, ori_mask)
        # Encode
        train_mask = encode_labels(train_mask)
        sample = [train_img.copy(), train_mask.copy()]
        if self.transform:
            sample = self.transform(sample)
        return sample


# pixel augmentation
class ImageAug(object):
    def __call__(self, sample):
        #sample对应的是getitem中的sample，取出来对应image和mask
        image, mask = sample

        if np.random.uniform(0,1) > 0.5: #np.random.uniform(0,1)在0-1之间随机取值

            #调整参数到还能看的出来语义信息为止，或者参考效果比较好的别人的方法，或者采取比较保守的，改变小的值
            seq = iaa.Sequential([iaa.OneOf([
                iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255)), #加一个高斯噪音 noises
                iaa.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.3)), #锐化
                iaa.GaussianBlur(sigma=(0, 1.0))])]) #高斯模糊进行随机选择

            image = seq.augment_image(image) #对图像进行增强方式的处理
        return image, mask


# deformation augmentation
class DeformAug(object):
    def __call__(self, sample):
        image, mask = sample
        seq = iaa.Sequential([iaa.CropAndPad(percent=(-0.05, 0.1))])
        seg_to = seq.to_deterministic()
        image = seg_to.augment_image(image)
        mask = seg_to.augment_image(mask)
        return image, mask

class DeformAug1(object)
    def __call__(self,sample):
        image,mask = sample
        aug = albu.RandomCrop(1000,360,p=0.5) #1000代表高度，360代表宽度，p是概率
        image,mask = aug(image=image,mask=mask)
        return image,mask

class ScaleAug(object):
    def __call__(self, sample):
        image, mask = sample
        scale = random.uniform(0.7, 1.5)
        h, w, _ = image.shape
        aug_image = image.copy()
        aug_mask = mask.copy()

        #对我们的image和mask进行缩放处理
        aug_image = cv2.resize(aug_image, (int (scale * w), int (scale * h)))
        aug_mask = cv2.resize(aug_mask, (int (scale * w), int (scale * h)))

        if (scale < 1.0):
            new_h, new_w, _ = aug_image.shape
            pre_h_pad = int((h - new_h) / 2)
            pre_w_pad = int((w - new_w) / 2)
            pad_list = [[pre_h_pad, h - new_h - pre_h_pad], [pre_w_pad, w - new_w - pre_w_pad], [0, 0]]
            aug_image = np.pad(aug_image, pad_list, mode="constant")
            aug_mask = np.pad(aug_mask, pad_list[:2], mode="constant")
        if (scale > 1.0):
            new_h, new_w, _ = aug_image.shape
            pre_h_crop = int ((new_h - h) / 2)
            pre_w_crop = int ((new_w - w) / 2)
            post_h_crop = h + pre_h_crop
            post_w_crop = w + pre_w_crop
            aug_image = aug_image[pre_h_crop:post_h_crop, pre_w_crop:post_w_crop]
            aug_mask = aug_mask[pre_h_crop:post_h_crop, pre_w_crop:post_w_crop]
        return aug_image, aug_mask


class CutOut(object):
    def __init__(self, mask_size, p):
        self.mask_size = mask_size
        self.p = p

    def __call__(self, sample):
        image, mask = sample

        mask_size_half = self.mask_size // 2
        offset = 1 if self.mask_size % 2 == 0 else 0

        h, w = image.shape[:2]

        #找到mask的中心位置
        cxmin, cxmax = mask_size_half, w + offset - mask_size_half
        cymin, cymax = mask_size_half, h + offset - mask_size_half

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)

        xmin, ymin = cx - mask_size_half, cy - mask_size_half #左上角的点
        xmax, ymax = xmin + self.mask_size, ymin + self.mask_size #右下角的点

        xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(w, xmax), min(h, ymax)

        if np.random.uniform(0, 1) < self.p:
            image[ymin:ymax, xmin:xmax] = (0, 0, 0)
        return image, mask


class ToTensor(object):
    def __call__(self, sample):

        image, mask = sample
        image = np.transpose(image,(2,0,1))
        image = image.astype(np.int32)
        mask = mask.astype(np.uint8)
        return {'image': torch.from_numpy(image.copy()),
                'mask': torch.from_numpy(mask.copy())}


def expand_resize_data(prediction=None, submission_size=(3384, 1710), offset=690):
    pred_mask = decode_labels(prediction)
    expand_mask = cv2.resize(pred_mask, (submission_size[0], submission_size[1] - offset), interpolation=cv2.INTER_NEAREST)
    submission_mask = np.zeros((submission_size[1], submission_size[0]), dtype='uint8')
    submission_mask[offset:, :] = expand_mask
    return submission_mask


def expand_resize_color_data(prediction=None, submission_size=(3384, 1710), offset=690):
    color_pred_mask = decode_color_labels(prediction)
    color_pred_mask = np.transpose(color_pred_mask, (1, 2, 0))
    color_expand_mask = cv2.resize(color_pred_mask, (submission_size[0], submission_size[1] - offset), interpolation=cv2.INTER_NEAREST)
    color_submission_mask = np.zeros((submission_size[1], submission_size[0], 3), dtype='uint8')
    color_submission_mask[offset:, :, :] = color_expand_mask
    return color_submission_mask
