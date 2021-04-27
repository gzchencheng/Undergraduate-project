#拓展自编码器
import torch.nn as nn
import torchvision
import AutoEncoder
class EAE(nn.Module):
    def __init__(self, input_size, emodel_name,pretrained=True):
        super(EAE, self).__init__()
        self.input_size = input_size
        self.emodel_name = emodel_name
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
        self.imodel = AutoEncoder.AutoEncoder(input_size=1000)

    def forward(self, x):
        img = self.emodel(x)
        encode,decode = self.imodel(img)
        return img, encode, decode


