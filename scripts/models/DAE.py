#两个输出的自编码器，每个自编码器都会输出两个模态的数据
import torch.nn as nn
class DAE(nn.Module):
    def __init__(self,input_size,output_size1,output_size2,feature_size=256):
        super(DAE, self).__init__()
        self.input_size = input_size
        self.output_size1 = output_size1
        self.output_size2 = output_size2
        self.feature_size = feature_size
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size,2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,self.feature_size),
            nn.ReLU(inplace=True)
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(self.feature_size,512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, self.output_size1),
            nn.Sigmoid()
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(self.feature_size,512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, self.output_size2),
            nn.Sigmoid()
        )

    def forward(self,x):
        encode = self.encoder(x)
        decode1 = self.decoder1(encode)
        decode2 = self.decoder2(encode)

        return encode,decode1,decode2


