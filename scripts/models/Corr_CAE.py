#跨模态的卷积自编码器
import AutoEncoder,torch,CAE
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import time,os,sys
import torch.utils.data as Data
import itertools
import heapq
from tqdm import tqdm, trange
import random
class Corr_CAE():
    def __init__(self, text_size: object, img_size: object,feature_size=256) -> object:
        self.text_size = text_size
        self.img_size = img_size
        self.feature_size = feature_size
        self.text_model = AutoEncoder.AutoEncoder(input_size=text_size,feature_size=feature_size)
        self.img_model = CAE.CAE()
        if torch.cuda.is_available():
            try:
                self.img_model = self.img_model.cuda()
                self.text_model = self.text_model.cuda()
            except:
                pass

    def train(self,texts,imgs,learning_rate=0.001,EPOCH=20,alpha=0.2,num_workers=4,batch_size=32,pin_memory=True):
        self.loss = []
        self.text_model.train()
        self.img_model.train()
        data_set = Data.TensorDataset(texts,imgs)
        data_loader = Data.DataLoader(dataset=data_set,num_workers=num_workers,shuffle=True,batch_size=batch_size,pin_memory=pin_memory)
        optimizer = optim.Adam(params=itertools.chain(self.text_model.parameters(), self.img_model.parameters()),
                               lr=learning_rate)
        loss_func = nn.MSELoss().cuda()
        for epoch in range(EPOCH):
            begin = time.time()
            eloss = 0
            i = 0
            for step, data in enumerate(data_loader):
                i += 1
                x, y = data
                x = Variable(x.cuda())
                y = Variable(y.cuda())
                x_encode, x_decode = self.text_model(x)
                y_encode, y_decode = self.img_model(y)
                loss = alpha * loss_func(x_encode,y_encode) + (1 - alpha) * (loss_func(x, x_decode) + loss_func(y, y_decode))
                eloss += loss.cpu().detach().numpy()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("epoch:{e} running time: {n}".format(e=str(epoch), n=str(time.time() - begin)))
            self.loss.append(eloss / i)
    #模型训练
    def save(self):
        try:
            os.mkdir('trained_models/')
            os.mkdir('trained_models/Corr_CAE/')
            path = 'trained_models/Corr_CAE/'
        except:
            path = 'trained_models/Corr_CAE/'
        torch.save(self.text_model,path+"text_model.model")
        torch.save(self.img_model,path+"img_model.model")
        f = open(path+'loss.txt','w')
        for loss in self.loss:
            f.write(str(loss))
            f.write('\n')
    #保存模型及训练损失
    def load(self,path = 'trained_models/Corr_CAE/'):
        try:
            self.text_model = torch.load(path+"text_model.model")
            self.img_model = torch.load(path+"img_model.model")
        except:
            path = input("input path of Corr_CAE:\n")
            self.text_model = torch.load(path + "text_model.model")
            self.img_model = torch.load(path + "img_model.model")
    #加载模型，默认将模型放在项目文件夹下，否则需要自行输入模型位置
    def predict(self,image=None,text=None):
        from data_processing import img_feature_get,text_feature_get
        if torch.cuda.is_available():
            self.text_model.cuda()
            self.img_model.cuda()
        else:
            self.text_model.cpu()
            self.img_model.cpu()
        img_encode, img_decode = ([],[])
        text_encode, text_decode = ([],[])
        if image:
            a = img_feature_get.get_img_feature([image])[0]
            if torch.cuda.is_available():
                a = a.cuda()
            img_encode,img_decode = self.img_model(a)
        if text:
            a = text_feature_get.get_text_feature([text])[0]
            if torch.cuda.is_available():
                a = a.cuda()
            text_encode,text_decode = self.text_model(a)

        if image and text:
            return  img_encode.detach().cpu(),text_encode.detach().cpu()
        if image:
            return img_encode.detach().cpu()
        if text:
            return text_encode.detach().cpu()
    #将输入的数据映射到公共表示区间
    def get_similarity(self,text,img):
        loss_func = nn.MSELoss()
        _img = img.unsqueeze(0)
        text_encode,text_decode = self.text_model(text)
        img_encode,img_decode = self.img_model(_img)
        return -loss_func(img_encode[0],text_encode).detach().cpu().numpy()#由于loss越高数据越不相关，因此使用loss的负数作为相关性
    #获取两个输入数据在公共表示空间内的相似性
    def get_samemodal_similarity(self,a,b,type):
        loss_func = nn.MSELoss().cuda()
        if type == 'text':
            a_encode, a_decode = self.text_model(a)
            b_encode, b_decode = self.text_model(b)
        else:
            _a = a.unsqueeze(0)
            _b = b.unsqueeze(0)
            a_encode, a_decode = self.img_model(_a)
            b_encode, b_decode = self.img_model(_b)
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