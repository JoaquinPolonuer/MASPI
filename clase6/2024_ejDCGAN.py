# ejemplo de https://github.com/pytorch/examples

from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from models import DiscriminatorDetalle, GeneratorDetalle, weights_init


class LPRdataset(Dataset):
    def __init__(self, root_path, transform):
        self.root_path = root_path
        self.transform = transform
        self.images = []

        for k in range(10):
            img_digits = sorted(
                [os.path.join(root_path, "%d" % k, i) for i in os.listdir(root_path + "/%d/" % k)]
            )
            self.images.extend(img_digits)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        folders = os.path.dirname(self.images[index])
        label = folders.split(os.path.sep)[-1:][0]

        return self.transform(img), int(label)

    def __len__(self):

        return len(self.images)


outf = "out"
if os.path.exists(outf) == False:
    os.mkdir(outf)

manualSeed = 1000
imageSize = (32, 15)
batchSize = 16

print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

cudnn.benchmark = True

dataroot = "BaseOCR_MultiStyle"
if os.path.exists(dataroot) == False:
    os.mkdir(dataroot)

dataset = LPRdataset(
    dataroot,
    transform=transforms.Compose(
        [
            transforms.Resize(imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
)

assert dataset
dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=0)

nc = 3
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda:0" if torch.cuda.is_available() else "cpu")
)
nz = 100
ngf = 32
ndf = 32

############################################################################
netG = GeneratorDetalle(nz, ngf, nc).to(device)
netG.apply(weights_init)
print(netG)

netD = DiscriminatorDetalle(ndf, nc).to(device)
netD.apply(weights_init)
print(netD)
############################################################################

criterion = nn.BCELoss()

fixed_noise = torch.randn(batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
lr = 0.002
beta1 = 0.5
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

netG.train()
netD.train()

niter = 50

for epoch in range(niter):
    with torch.no_grad():
        fake = netG(fixed_noise)
    vutils.save_image(
        fake.detach(), "%s/fake_samples_epoch_%03d.png" % (outf, epoch), normalize=True
    )

    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        ########################
        # completar aqui
        # declarar como vector torch y completar con el target correcto.
        # cuidado las dimensiones

        label = torch.ones(data[0].shape[0], device=real_cpu.device)
        ########################
        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        ########################
        # completar aqui
        # declarar como vector torch y completar con el target correcto.
        # cuidado las dimensiones
        label = torch.zeros(data[0].shape[0], device=real_cpu.device)
        ########################
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        ########################
        # completar aqui
        # declarar como vector torch y completar con el target correcto.
        # cuidado las dimensiones
        label = torch.ones(data[0].shape[0], device=real_cpu.device)
        ########################
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print(
            "[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f"
            % (epoch, niter, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2)
        )

vutils.save_image(real_cpu, "%s/real_samples.png" % outf, normalize=True)
