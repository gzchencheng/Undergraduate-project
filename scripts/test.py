#MNIST数据集测试
import torch,torchvision,fastai
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import time,os
import torch.utils.data as Data
import itertools
from models import Corr_AE,ECAE,CFAE,CAE,Corr_CAE
import cv2 as cv
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def get_loss(a,b):
    a = a.detach().cpu().numpy()
    b = b.detach().cpu().numpy()
    result = 0
    for i in range(len(a)):
        result += (a[i]-b[i])**2
    return result/len(a)

transform = transforms.Compose([
    transforms.Resize((100,100)),
    transforms.ToTensor()
])

class my_loss(nn.Module):
    def __init__(self):
        super(my_loss,self).__init__()
    def forward(self,x,y):
        if len(x.shape) == 1:
            return 1 - torch.abs(torch.sum(x*y))
        return 1 - torch.mean(torch.abs(torch.sum(x*y,dim=1)))

if __name__ == '__main__':
    f = open('F:/test.txt','r')
    print(f.read())
    '''train_data = torchvision.datasets.MNIST(root='F:/python/mnist', train=True)
    train_x = []
    for i in tqdm(range(10000)):
        img = cv.cvtColor(cv.resize(cv.imread('F:/mnist/train/' + str(i) + '.png'), (250, 250)), cv.COLOR_BGR2GRAY)
        train_x.append(img)
    train_x = torch.tensor(train_x).unsqueeze_(1).float() / 255
    _train_y = train_data.targets.numpy()
    train_y = []
    for y in _train_y:
        temp = [0] * 3000
        temp[y * 300] = 1
        train_y.append(temp)
    train_y = torch.tensor(train_y).float()[:10000]

    test_data = torchvision.datasets.MNIST(root='F:/python/mnist', train=False)
    test_x = []
    for i in tqdm(range(1000)):
        img = cv.cvtColor(cv.resize(cv.imread('F:/mnist/test/' + str(i) + '.png'), (250, 250)), cv.COLOR_BGR2GRAY)
        test_x.append(img)
    test_x = torch.tensor(test_x).unsqueeze_(1).float() / 255
    test_y = test_data.targets.numpy()[:1000]

    model = Corr_CAE.Corr_CAE(img_size=250 * 250, text_size=3000)
    model.train(train_y, train_x, EPOCH=200, alpha=0.2, num_workers=4, batch_size=64)
    print(model.loss)

    accuracy = 0
    for i in tqdm(range(len(test_x))):
        x = test_x[i:i + 1].cuda()
        encode, decode = model.img_model(x)
        encode = encode[0]
        result = 0
        min_loss = 1000
        for j in range(10):
            temp = [0] * 3000
            temp[j * 300] = 1
            a, b = model.text_model(torch.tensor(temp).float().cuda())
            loss = get_loss(a, encode)
            if loss < min_loss:
                min_loss = loss
                result = j
        if result == test_y[i]:
            accuracy += 1
    print("accuracy:", accuracy / len(test_x))

    while 1:
        i = int(input("i:"))
        if i == -1:
            break
        x = train_x[i][0]
        plt.imshow(x,cmap='gray')
        plt.show()

        encode,decode = model.img_model(train_x[i:i+1].cuda())
        y = decode.detach().cpu().numpy()[0][0]
        plt.imshow(y,cmap='gray')
        plt.show()'''