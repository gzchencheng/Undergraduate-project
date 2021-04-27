#将图像转化为250*300维的向量，进行归一化后返回
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
import torch
from albumentations import CLAHE,GaussianBlur,RandomGamma
from tqdm import tqdm, trange
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def img_enhance(img):
    gauss = GaussianBlur()
    img = gauss.apply(img,ksize=5)
    #高斯模糊
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    clahe = CLAHE()
    img = clahe.apply(img)
    #对比度受限的自适应直方图均衡化
    s = img.shape
    v = 0
    for i in range(s[0]):
        for j in range(s[1]):
            v += hsv[i][j][2]
    v = v / s[0] / s[1]
    if v > 120:
        gamma = RandomGamma()
        img = gamma.apply(img,gamma=1.2)
    elif v < 80:
        gamma = RandomGamma()
        img = gamma.apply(img, gamma=0.8)
    #如果图像过亮或过暗，则进行伽马变换

    return img
#图像增强

def img_load(file):
    img = cv.imread(file, cv.IMREAD_COLOR)
    '''img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = img_enhance(img)
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)'''
    #考虑到数据处理需要大量时间，因此原数据已经过增强处理并存储
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img = cv.resize(img,dsize=(250,300))
    return img


def get_img_feature(files,mode):
    result = []
    i = 1
    for file in files:
        if i % 1000 == 0:
            print(i, "images of", len(files), "images have been loaded")
        i += 1
        if mode == 1 or mode == 2:
            result.append(img_load(file))
        elif mode == 3:
            img = cv.cvtColor(cv.resize(cv.imread(file), (250, 250)), cv.COLOR_BGR2GRAY)
            result.append(img)
        else:
            result.append(transform(Image.open(file)).numpy())
    if mode == 1 or mode == 2:
        return torch.tensor(result).view(-1,250*300).float()/255
    elif mode == 3:
        return torch.tensor(result).unsqueeze_(1).float()/255
    else:
        return torch.tensor(result).float()/255
#返回250*300的图像向量