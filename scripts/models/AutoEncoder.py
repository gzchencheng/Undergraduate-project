#自编码器
import torch.nn as nn
class AutoEncoder(nn.Module):
    def __init__(self,input_size,feature_size=256):
        super(AutoEncoder, self).__init__()
        self.input_size = input_size
        self.feature_size = feature_size
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size,2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,self.feature_size),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.feature_size,512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, self.input_size),
            nn.Sigmoid()
        )

    def forward(self,x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode,decode


