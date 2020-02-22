#####VAE model for 32 by 32 images,nb size of the image at bottleneck, with flatten bottleneck and notflatten n_f#####
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image


from random import randint

from IPython.display import Image
from IPython.core.display import Image, display


class VAE(nn.Module):
    def __init__(self, nc,latent_variable_size,nb):
        super(VAE, self).__init__()

        self.nc = nc
        self.nb=nb
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv2d(nc,64, kernel_size=5, stride=2,padding=(2,2))
        self.bn1 = nn.BatchNorm2d(64)

        self.e2 = nn.Conv2d(64,128, kernel_size=5, stride=2,padding=(2,2))
        self.bn2 = nn.BatchNorm2d(128)

        self.e3 = nn.Conv2d(128, 256,kernel_size=5, stride=2,padding=(2,2))
        self.bn3 = nn.BatchNorm2d(256)


        self.fc1 = nn.Linear(256*self.nb*self.nb, latent_variable_size)
        self.fc2 = nn.Linear(256*self.nb*self.nb, latent_variable_size)

        # decoder
        self.d1 = nn.Linear(latent_variable_size, 256*self.nb*self.nb)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(2)
        self.d2 = nn.Conv2d(256, 256,kernel_size=5)
        self.bn4 = nn.BatchNorm2d(256, 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(2)
        self.d3 = nn.Conv2d(256, 128,kernel_size=5)
        self.bn5 = nn.BatchNorm2d(128, 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(2)
        self.d4 = nn.Conv2d(128, 32,kernel_size=5)
        self.bn6 = nn.BatchNorm2d(32, 1.e-3)

        self.pd4 = nn.ReplicationPad2d(2)
        self.d5 = nn.Conv2d(32,self.nc,kernel_size=5)


        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.bn1(self.e1(x)))
        h2 = self.relu(self.bn2(self.e2(h1)))
        h3 = self.relu(self.bn3(self.e3(h2)))
        h3 = h3.view(-1, 256*self.nb*self.nb)

        return self.fc1(h3), self.fc2(h3)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, 256, self.nb, self.nb)
        #h2=self.up1(h1)
        #h2=self.pd1(h2)
        #h2=self.d2(h2)
        #h2=self.bn4(h2)
        h2 = self.relu(self.bn4(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.relu(self.bn5(self.d3(self.pd2(self.up2(h2)))))
        h4 = self.relu(self.bn6(self.d4(self.pd3(self.up3(h3)))))

        return self.sigmoid(self.relu(self.d5(self.pd4(h4))))

 

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar

        
class VAE_nf(nn.Module):
    def __init__(self, nc,latent_variable_size):
        super(VAE_nf, self).__init__()

        self.nc = nc
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv2d(nc,64, kernel_size=5, stride=2,padding=(2,2))
        self.bn1 = nn.BatchNorm2d(64)

        self.e2 = nn.Conv2d(64,128, kernel_size=5, stride=2,padding=(2,2))
        self.bn2 = nn.BatchNorm2d(128)

        self.e3 = nn.Conv2d(128, 256,kernel_size=5, stride=2,padding=(2,2))
        self.bn3 = nn.BatchNorm2d(256)

        self.e4 = nn.Conv2d(256, latent_variable_size,kernel_size=5, stride=2,padding=(2,2))
        self.e5 = nn.Conv2d(256, latent_variable_size,kernel_size=5, stride=2,padding=(2,2))

        # decoder
        self.up0 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd0 = nn.ReplicationPad2d(2)
        self.d1 = nn.Conv2d(latent_variable_size, 256,kernel_size=5)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(2)
        self.d2 = nn.Conv2d(256, 256,kernel_size=5)
        self.bn4 = nn.BatchNorm2d(256, 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(2)
        self.d3 = nn.Conv2d(256, 128,kernel_size=5)
        self.bn5 = nn.BatchNorm2d(128, 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(2)
        self.d4 = nn.Conv2d(128, 32,kernel_size=5)
        self.bn6 = nn.BatchNorm2d(32, 1.e-3)

        self.pd4 = nn.ReplicationPad2d(2)
        self.d5 = nn.Conv2d(32,self.nc,kernel_size=5)


        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.bn1(self.e1(x)))
        h2 = self.relu(self.bn2(self.e2(h1)))
        h3 = self.relu(self.bn3(self.e3(h2)))
        #h3 = h3.view(-1, 256*self.nb*self.nb)

        return self.e4(h3), self.e5(h3)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  

    def decode(self, z):
        h1 = self.relu(self.d1(self.pd0(self.up0(z))))
        h2 = self.relu(self.bn4(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.relu(self.bn5(self.d3(self.pd2(self.up2(h2)))))
        h4 = self.relu(self.bn6(self.d4(self.pd3(self.up3(h3)))))

        return self.sigmoid(self.relu(self.d5(self.pd4(h4))))

 

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar

        #####VAE model for 32 by 32 images,nb size of the image at bottleneck, with flatten bottleneck and notflatten n_f#####
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image


from random import randint

from IPython.display import Image
from IPython.core.display import Image, display


class VAE(nn.Module):
    def __init__(self, nc,latent_variable_size,nb):
        super(VAE, self).__init__()

        self.nc = nc
        self.nb=nb
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv2d(nc,64, kernel_size=5, stride=2,padding=(2,2))
        self.bn1 = nn.BatchNorm2d(64)

        self.e2 = nn.Conv2d(64,128, kernel_size=5, stride=2,padding=(2,2))
        self.bn2 = nn.BatchNorm2d(128)

        self.e3 = nn.Conv2d(128, 256,kernel_size=5, stride=2,padding=(2,2))
        self.bn3 = nn.BatchNorm2d(256)


        self.fc1 = nn.Linear(256*self.nb*self.nb, latent_variable_size)
        self.fc2 = nn.Linear(256*self.nb*self.nb, latent_variable_size)

        # decoder
        self.d1 = nn.Linear(latent_variable_size, 256*self.nb*self.nb)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(2)
        self.d2 = nn.Conv2d(256, 256,kernel_size=5)
        self.bn4 = nn.BatchNorm2d(256, 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(2)
        self.d3 = nn.Conv2d(256, 128,kernel_size=5)
        self.bn5 = nn.BatchNorm2d(128, 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(2)
        self.d4 = nn.Conv2d(128, 32,kernel_size=5)
        self.bn6 = nn.BatchNorm2d(32, 1.e-3)

        self.pd4 = nn.ReplicationPad2d(2)
        self.d5 = nn.Conv2d(32,self.nc,kernel_size=5)


        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.bn1(self.e1(x)))
        h2 = self.relu(self.bn2(self.e2(h1)))
        h3 = self.relu(self.bn3(self.e3(h2)))
        h3 = h3.view(-1, 256*self.nb*self.nb)

        return self.fc1(h3), self.fc2(h3)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, 256, self.nb, self.nb)
        #h2=self.up1(h1)
        #h2=self.pd1(h2)
        #h2=self.d2(h2)
        #h2=self.bn4(h2)
        h2 = self.relu(self.bn4(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.relu(self.bn5(self.d3(self.pd2(self.up2(h2)))))
        h4 = self.relu(self.bn6(self.d4(self.pd3(self.up3(h3)))))

        return self.sigmoid(self.relu(self.d5(self.pd4(h4))))

 

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar

        
class simple_VAE(nn.Module):
    def __init__(self, nc,latent_variable_size):
        super(simple_VAE, self).__init__()

        self.nc = nc
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv2d(nc,64, kernel_size=5, stride=2,padding=(2,2))
        self.bn1 = nn.BatchNorm2d(64)

        self.e4 = nn.Conv2d(64, latent_variable_size,kernel_size=5, stride=2,padding=(2,2))
        self.e5 = nn.Conv2d(64, latent_variable_size,kernel_size=5, stride=2,padding=(2,2))

        # decoder
        self.up0 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd0 = nn.ReplicationPad2d(2)
        self.d1 = nn.Conv2d(latent_variable_size, 64,kernel_size=5)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(2)
        self.d2 = nn.Conv2d(64, 1,kernel_size=5)
        #self.bn4 = nn.BatchNorm2d(256, 1.e-3)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.bn1(self.e1(x)))
        
        #h3 = h3.view(-1, 256*self.nb*self.nb)

        return self.e4(h1), self.e5(h1)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  

    def decode(self, z):
        h1 = self.relu(self.d1(self.pd0(self.up0(z))))
        h2 = self.relu(self.d2(self.pd1(self.up1(h1))))

        return self.sigmoid(self.relu(h2))

 

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar

class vanila_VAE(nn.Module):
    def __init__(self, nc,latent_variable_size):
        super(vanila_VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, latent_variable_size)
        self.fc22 = nn.Linear(400, latent_variable_size)
        self.fc3 = nn.Linear(latent_variable_size, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))


        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        #if torch.isnan(h3 ):
            #print(h3 )
        return torch.sigmoid(self.fc4(h3))
        if torch.isnan(torch.sigmoid(self.fc4(h3))):
            print(h3 )

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        #f sum(sum(torch.isnan(mu)))>0:
           #print(z )
        return self.decode(z), mu, logvar