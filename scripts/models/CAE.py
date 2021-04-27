#卷积自编码器
import torch.nn as nn
class CAE(nn.Module):
    def __init__(self,input_size=250*250):
        super(CAE, self).__init__()
        self.input_size = input_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 5, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(8, 1, 5, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 8, 8, 2),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 16, 8, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 7, 3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self,x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode.view(encode.shape[0],-1), decode