from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class DiscriminatorDetalle(nn.Module):
    def __init__(self, ndf, nc):
        super(DiscriminatorDetalle, self).__init__()

        self.cv0 = nn.Conv2d(nc, ndf, 4, (2, 1), (1, 0), bias=False)

        self.cv1 = nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False)
        self.bt1 = nn.BatchNorm2d(ndf * 2)

        self.cv2 = nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False)
        self.bt2 = nn.BatchNorm2d(ndf * 4)

        self.cv3 = nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 0, bias=False)
        self.bt3 = nn.BatchNorm2d(ndf * 8)

        self.cv4 = nn.Conv2d(ndf * 8, 1, (2, 1), 1, 0, bias=False)

        self.lek = nn.LeakyReLU(0.2, inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, input):
        x = self.cv0(input)
        x = self.lek(x)

        x = self.cv1(x)
        x = self.lek(self.bt1(x))

        x = self.cv2(x)
        x = self.lek(self.bt2(x))

        x = self.cv3(x)
        x = self.lek(self.bt3(x))

        x = self.cv4(x)
        return self.sig(x).view(x.size(0))


class GeneratorDetalle(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(GeneratorDetalle, self).__init__()

        self.ct0 = nn.ConvTranspose2d(nz, ngf * 8, 2, 1, 0, bias=False)
        self.bt0 = nn.BatchNorm2d(ngf * 8)

        self.ct1 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.bt1 = nn.BatchNorm2d(ngf * 4)

        self.ct2 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.bt2 = nn.BatchNorm2d(ngf * 2)

        self.ct3 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.bt3 = nn.BatchNorm2d(ngf)

        self.ct4 = nn.ConvTranspose2d(ngf, nc, (4, 2), (2, 1), 1, bias=False)

        self.elu = nn.ELU(alpha=0.1, inplace=True)
        self.tan = nn.Tanh()

    def forward(self, input):

        x = self.ct0(input)
        x = self.elu(self.bt0(x))

        x = self.ct1(x)
        x = self.elu(self.bt1(x))

        x = self.ct2(x)
        x = self.elu(self.bt2(x))

        x = self.ct3(x)
        x = self.elu(self.bt3(x))

        x = self.ct4(x)

        return self.tan(x)
