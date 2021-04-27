#训练对应自编码器模型
import torch,torchvision
import matplotlib.pyplot as plt
from models import Corr_CAE
import csv
from data_processing import img_feature_get,text_feature_get
from multiprocessing import cpu_count
import threading
import time,os
import cv2 as cv
n = 2#信号量，保证数据读取完毕后再进行模型的训练

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
        global _img_data,img_data
        img_data = img_feature_get.get_img_feature(files=_img_data, mode=3)
        #图像数据归一化
        global n
        n -= 1
#图像信息读取线程

if __name__ == '__main__':
    x = input("input path of data:")
    _text_data = []
    _img_data = []
    texts = list(csv.reader(open(x+'/cxr/report/indiana_reports.csv',encoding='utf-8')))[1:]
    texts = {texts[i][0]:texts[i][6] for i in range(len(texts)) if texts[i][6] != ""}
    imgs = list(csv.reader(open(x+'/cxr/report/indiana_projections.csv',encoding='utf-8')))[1:]

    for i in range(len(imgs)):
        uid = imgs[i][0]
        filename = 'CXR' + imgs[i][1].replace('.dcm','')
        if uid in texts:
            _text_data.append(texts[uid])
            _img_data.append(x + '/cxr/image/' + filename)
            _text_data.append(texts[uid])
            _img_data.append(x+'/cxr/image/'+'flip_'+filename)

    text_data = []
    img_data = []
    t1 = TextLoaderThread()
    t2 = ImgLoaderThread()
    t1.start()
    t2.start()
    #通过两个线程同时对图像数据和文本数据
    while n:
        time.sleep(5)
    #每5秒主线程检查数据是否读取完毕


    model = Corr_CAE.Corr_CAE(len(text_data[0]),len(img_data[0]))
    x = int(input("1.model train\t2.model evaluate\n"))
    if x == 1:
        for i in range(10):
            model.train(text_data, img_data, batch_size=64, num_workers=cpu_count(), EPOCH=200, alpha=0.2)
            model.save()
    # 模型的训练与存储
    else:
        model.load()
        f = open('./Corr_CAE_topkacc.txt', 'a')
        f.write(str(model.GetTopkAccuracy(texts=text_data.cuda(), imgs=img_data.cuda(), k=int(0.2 * len(text_data)),
                                          search_mode=1)))
        f.flush()
        f.write('\t')
        f.write(str(model.GetTopkAccuracy(texts=text_data.cuda(), imgs=img_data.cuda(), k=int(0.2 * len(text_data)),
                                          search_mode=2)))
        f.flush()
        f.write('\t')
        for beta in [0.3, 0.5, 0.7]:
            f.write(str(model.GetTopkAccuracy(texts=text_data.cuda(), imgs=img_data.cuda(), k=int(0.2 * len(text_data)),
                                              search_mode=3, beta=beta)))
            f.flush()
            f.write('\t')
        f.write(str(model.GetTopkAccuracy(texts=text_data.cuda(), imgs=img_data.cuda(), k=int(0.2 * len(text_data)),
                                          search_mode=4, beta=0.5)))
        f.flush()
        f.write('\t')
        f.write('\n')
    #模型评价

