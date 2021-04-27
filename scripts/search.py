#脚本化检索系统
from models import Corr_AE,Corr_CAE,CFAE,ECAE
import torch,torchvision
import matplotlib.pyplot as plt
import csv
from data_processing import img_feature_get,text_feature_get
from multiprocessing import cpu_count
import threading
import time,os
import cv2 as cv
from PIL import Image
n = 2#信号量，保证数据读取完毕后再进行模型的训练
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

class TextLoaderThread(threading.Thread):
    def __init__(self):
        super(TextLoaderThread,self).__init__()
    def run(self):
        global _text_data,text_data
        text_data = text_feature_get.get_text_feature(texts=_text_data)
        mi = text_data.min().numpy()
        ma = text_data.max().numpy()
        text_data = (text_data - mi) / (ma - mi)
        #文本数据0-1归一化
        global n
        n -= 1
#文本信息读取线程

class ImgLoaderThread(threading.Thread):
    def __init__(self):
        super(ImgLoaderThread,self).__init__()
    def run(self):
        global _img_data,img_data,y
        img_data = img_feature_get.get_img_feature(files=_img_data[:5000],mode=y)
        global n
        n -= 1
#图像信息读取线程


if __name__ == '__main__':
    x = input("input path of data:")
    y = int(input("select model: 1.Corr_AE  2.CFAE  3.Corr_CAE  4.ECAE\n"))
    texts = list(csv.reader(open(x + '/cxr/report/indiana_reports.csv', encoding='utf-8')))[1:]
    _text_data = [texts[i][6] for i in range(len(texts)) if texts[i][6] != ""]
    imgs = list(csv.reader(open(x + '/cxr/report/indiana_projections.csv', encoding='utf-8')))[1:]

    _img_data = []
    for i in range(len(imgs)):
        filename = 'CXR' + imgs[i][1].replace('.dcm', '')
        _img_data.append(x + '/cxr/image/' + filename)
        _img_data.append(x + '/cxr/image/' + 'flip_' + filename)

    text_data = []
    img_data = []
    t1 = TextLoaderThread()
    t2 = ImgLoaderThread()
    t1.start()
    t2.start()
    # 通过两个线程同时对图像数据和文本数据
    while n:
        time.sleep(5)
    # 每5秒主线程检查数据是否读取完毕

    if y == 1:
        model = Corr_AE.Corr_AE(text_size=len(text_data[0]),img_size=len(img_data[0]))
    elif y == 2:
        model = CFAE.CFAE(len(text_data[0]), len(img_data[0]))
    elif y == 3:
        model = Corr_CAE.Corr_CAE(text_size=len(text_data[0]),img_size=250*250)
    else:
        model = ECAE.ECAE(text_size=len(text_data[0]), img_size=len(img_data[0]),model_name='resnet101')
    model.load()
    while True:
        mode = int(input('search mode:1.img2text 2.text2img 3.img_text2text 4.img_text2img 5.exit\n'))
        if mode == 5:
            break
        data = input("input data:\n")
        if mode == 1:
            result = model.search_top3(mode=mode, search_data=text_data,
                                       img=img_feature_get.get_img_feature([data], mode=y)[0])
        elif mode == 2:
            result = model.search_top3(mode=mode, search_data=img_data,
                                       text=text_feature_get.get_text_feature([data])[0])
        elif mode == 3:
            _data = input()
            result = model.search_top3(mode=mode, search_data=text_data,
                                       img=img_feature_get.get_img_feature([data], mode=y)[0],
                                       text=text_feature_get.get_text_feature([_data])[0])
        else:
            _data = input()
            result = model.search_top3(mode=mode, search_data=img_data,
                                       img=img_feature_get.get_img_feature([data], mode=y)[0],
                                       text=text_feature_get.get_text_feature([_data])[0])
        print("result:")
        for i in result:
            if mode % 2 == 1:
                print(_text_data[i])
            else:
                print(_img_data[i])
