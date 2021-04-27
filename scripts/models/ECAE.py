#通过迁移学习实现对图像数据特征的提取，并搭建拓展的跨模态自编码器
import AutoEncoder,EAE
import torch,torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import time,os,sys
import torch.utils.data as Data
import itertools
from tqdm import tqdm
import heapq
import random

class ECAE():
    def __init__(self,text_size,model_name,img_size,pretrained=True):
        self.text_size = text_size
        self.img_size = img_size
        self.text_model = AutoEncoder.AutoEncoder(self.text_size)
        self.img_model = AutoEncoder.AutoEncoder(input_size=text_size)
        self.vec = nn.Sequential(
            nn.Linear(1000,text_size),
            nn.ReLU(inplace=True),
        )
        self.emodel_name = model_name
        if self.emodel_name == "vgg19":
            self.emodel = torchvision.models.vgg19(pretrained=pretrained)
        elif self.emodel_name == "alexnet":
            self.emodel = torchvision.models.alexnet(pretrained=pretrained)
        elif self.emodel_name == "densenet161":
            self.emodel = torchvision.models.densenet161(pretrained=pretrained)
        elif self.emodel_name == "resnet101":
            self.emodel = torchvision.models.resnet101(pretrained=pretrained)
        elif self.emodel_name == "squeezenet1_0":
            self.emodel = torchvision.models.squeezenet1_0(pretrained=pretrained)
        elif self.emodel_name == "inception_v3":
            self.emodel = torchvision.models.inception_v3(pretrained=pretrained)
        if torch.cuda.is_available():
            try:
                self.img_model = self.img_model.cuda()
                self.text_model = self.text_model.cuda()
                self.emodel = self.emodel.cuda()
                self.vec = self.vec.cuda()
            except:
                pass
        self.loss = []

    def train(self, texts, imgs, learning_rate=0.001, EPOCH=20, alpha=0.2, num_workers=4, batch_size=32,
              pin_memory=True):
        data_set = Data.TensorDataset(texts, imgs)
        data_loader = Data.DataLoader(dataset=data_set, num_workers=num_workers, shuffle=True, batch_size=batch_size,
                                      pin_memory=pin_memory)
        optimizer = optim.AdamW(params=itertools.chain(self.text_model.parameters(), self.img_model.parameters(), self.emodel.parameters()),
                               lr=learning_rate)
        loss_func = nn.MSELoss().cuda()

        for epoch in range(EPOCH):
            begin = time.time()
            eloss = 0
            i = 0
            for step,data in enumerate(data_loader):
                i += 1
                text, img = data
                text = Variable(text.cuda())
                img = Variable(img.cuda())
                a = self.emodel(img)
                img_feature = self.vec(a)
                img_encode, img_decode = self.img_model(img_feature)
                text_encode, text_decode = self.text_model(text)
                loss =loss_func(img_feature,text) + alpha * loss_func(img_encode, text_encode) + (1 - alpha) * (
                            loss_func(img_feature, img_decode) + loss_func(text, text_decode))
                eloss += loss.cpu().detach().numpy()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("epoch:{e} running time: {n}".format(e=str(epoch), n=str(time.time() - begin)))
            self.loss.append(eloss/i)
    def save(self):
        try:
            os.mkdir('trained_models/')
        except:
            pass
        try:
            os.mkdir('trained_models/Extended_Corr_AE/')
        except:
            pass
        try:
            os.mkdir('trained_models/Extended_Corr_AE/{e}_Corr_AE/'.format(e=self.emodel_name))
        except:
            pass
        path = 'trained_models/Extended_Corr_AE/{e}_Corr_AE/'.format(e=self.emodel_name)
        torch.save(self.text_model,path+"text_model.model")
        torch.save(self.img_model,path+"img_model.model")
        torch.save(self.emodel,path+self.emodel_name+".model")
        f = open(path+'loss.txt','w')
        for loss in self.loss:
            f.write(str(loss))
            f.write('\n')
    def load(self,path = 'trained_models/Extended_Corr_AE/'):
        try:
            self.text_model = torch.load(path+"{e}_Corr_AE/text_model.model".format(e=self.emodel_name))
            self.img_model = torch.load(path+"{e}_Corr_AE/img_model.model".format(e=self.emodel_name))
            self.emodel = torch.load(path+"{e}_Corr_AE/{e}.model".format(e=self.emodel_name))
        except:
            path = input("input path of Corr_AE:\n")
            self.text_model = torch.load(path + "{e}_Corr_AE/text_model.model".format(e=self.emodel_name))
            self.img_model = torch.load(path + "{e}_Corr_AE/img_model.model".format(e=self.emodel_name))
            self.emodel = torch.load(path + "{e}_Corr_AE/{e}.model".format(e=self.emodel_name))
    def get_similarity(self,text,img):
        loss_func = nn.MSELoss()
        _img = img.unsqueeze(0)
        text_encode,text_decode = self.text_model(text)
        img_feature = self.vec(self.emodel(_img))
        img_encode,img_decode = self.img_model(img_feature[0])
        return -loss_func(img_encode,text_encode).detach().cpu().numpy()#由于loss越高数据越不相关，因此使用loss的负数作为相关性
    #获取两个输入数据在公共表示空间内的相似性
    def get_samemodal_similarity(self,a,b,type):
        loss_func = nn.MSELoss().cuda()
        if type == 'text':
            a_encode, a_decode = self.text_model(a)
            b_encode, b_decode = self.text_model(b)
        else:
            _a = a.unsqueeze(0)
            _b = b.unsqueeze(0)
            a_feature = self.vec(self.emodel(_a))
            b_feature = self.vec(self.emodel(_b))
            a_encode, a_decode = self.img_model(a_feature[0])
            b_encode, b_decode = self.img_model(b_feature[0])
        return -loss_func(a_encode, b_encode).detach().cpu().numpy()
    #获取同模态数据在公共表示空间内的相似性
    def GetTopkAccuracy(self, texts, imgs, k,search_mode,beta=0.5):
        n = int(len(texts) / 400)  # 从所有数据中选取0.25%作为测试数据
        start = random.randint(0, len(texts) - n)
        acc = 0
        for i in tqdm(range(start, start + n)):
            min_heap = []  # 用于获取相似度前k大数据的最小堆
            img = imgs[i]
            text = texts[i]
            isTopk = False
            for j in range(len(texts)):
                if search_mode == 1:
                    similarity = self.get_similarity(text=texts[j], img=img)
                #图检文
                elif search_mode == 2:
                    similarity = self.get_similarity(text=text, img=imgs[j])
                #文检图
                elif search_mode == 3:
                    similarity = beta * self.get_similarity(text=texts[j], img=img) + (1-beta) * self.get_samemodal_similarity(text,texts[j],type='text')
                #图文检文
                else:
                    similarity = beta * self.get_similarity(text=text, img=imgs[j]) + (1 - beta) * self.get_samemodal_similarity(img, imgs[j], type='img')
                #图文检图
                # 根据不同任务获取数据相似度
                if len(min_heap) < k:
                    if j == i:
                        isTopk = True
                    heapq.heappush(min_heap, (similarity, j))
                # 当堆的大小小于k时，数据直接进堆
                else:
                    if min_heap[0][1] == i and min_heap[0][0] < similarity:
                        isTopk = False
                        break
                    # 如果被替换的数据为测试数据，则检索失败
                    if j == i and similarity > min_heap[0][0]:
                        isTopk = True
                    heapq.heappushpop(min_heap, (similarity, j))
                    # 数据进堆
            if isTopk:
                acc += 1
        return acc / n
    def search_top3(self,mode,search_data,beta=0.5,img=None,text=None):
        '''
        :param mode: 1.图检文，2.文检图，3.联合检文，4.联合检图
        :param img:
        :param text:
        :param search_data: 欲检索数据
        :return:
        '''
        min_heap = []
        result = []
        if img != None:
            img = img.cuda()
        if text != None:
            text = text.cuda()
        search_data = search_data.cuda()
        for i in range(len(search_data)):
            data = search_data[i]
            if mode == 1:
                similarity = self.get_similarity(text=data, img=img)
            elif mode == 2:
                similarity = self.get_similarity(text=text, img=data)
            elif mode == 3:
                similarity = beta * self.get_similarity(text=data, img=img) + (
                            1 - beta) * self.get_samemodal_similarity(data, text, type='text')
            else:
                similarity = beta * self.get_similarity(img=data, text=text) + (
                            1 - beta) * self.get_samemodal_similarity(data, img, type='img')
            if len(min_heap) < 3:
                heapq.heappush(min_heap, (similarity, i))
            else:
                heapq.heappushpop(min_heap, (similarity, i))
        for h in min_heap:
            result.append(h[1])
        return result