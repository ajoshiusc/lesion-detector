import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.utils as vutils




##########define network##########
class Encoder(nn.Module):
    def __init__(self, input_channels, output_channels, representation_size = 32):
        super(Encoder, self).__init__()
        # input parameters
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        self.features = nn.Sequential(
            # nc x 8x 8
            nn.Conv2d(self.input_channels, representation_size, 5, stride=2, padding=2),
            nn.BatchNorm2d(representation_size),
            nn.ReLU(),
            # hidden_size x 4 x 4
            nn.Conv2d(representation_size, representation_size*2, 5, stride=2, padding=2),
            nn.BatchNorm2d(representation_size * 2),
            nn.ReLU(),
            # hidden_size*2 x 2x 2
        )
            
        self.mean = nn.Sequential(
            nn.Linear(representation_size*2*2*2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_channels))
        
        self.logvar = nn.Sequential(
            nn.Linear(representation_size*2*2*2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_channels))
        
    def forward(self, x):
        batch_size = x.size()[0]

        hidden_representation = self.features(x)

        mean = self.mean(hidden_representation.view(batch_size, -1))
        logvar = self.logvar(hidden_representation.view(batch_size, -1))

        return mean, logvar
    
    def hidden_layer(self, x):
        batch_size = x.size()[0]
        output = self.features(x)
        return output

class Decoder(nn.Module):
    def __init__(self, input_size, representation_size):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.representation_size = representation_size
        dim = representation_size[0] * representation_size[1] * representation_size[2]
        
        self.preprocess = nn.Sequential(
            nn.Linear(input_size, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU())
        
            # 256 x 2 x 2
        self.deconv1 = nn.ConvTranspose2d(representation_size[0], 32, 5, stride=2, padding=3)
        self.act1 = nn.Sequential(nn.BatchNorm2d(32),
                                  nn.ReLU())
            # 256 x 4 x 4
        self.deconv2 = nn.ConvTranspose2d(32, 16, 5, stride=2, padding=2)
        self.act2 = nn.Sequential(nn.BatchNorm2d(16),
                                  nn.ReLU())
            # 128 x 8 x 8
        self.deconv3 = nn.ConvTranspose2d(16, 16, 5, stride=2, padding=2)
        self.act3 = nn.Sequential(nn.BatchNorm2d(16),
                                  nn.ReLU())
            # 32 x 128 x 128
        self.deconv4 = nn.ConvTranspose2d(16, 3, 5, stride=1, padding=2)
        self.deconv5 = nn.ConvTranspose2d(16, 3, 5, stride=1, padding=2)
            # 3 x 128 x 128
        self.activation = nn.Tanh()
        self.relu=nn.ReLU()
        self.activation2=nn.Sigmoid()    
    
    def forward(self, code):
        bs = code.size()[0]
        preprocessed_codes = self.preprocess(code)
        preprocessed_codes = preprocessed_codes.view(-1,
                                                     self.representation_size[0],
                                                     self.representation_size[1],
                                                     self.representation_size[2])
        output = self.deconv1(preprocessed_codes, output_size=(bs, 32, 2, 2))
        output = self.act1(output)
        output = self.deconv2(output, output_size=(bs, 16, 4, 4))
        output = self.act2(output)
        output = self.deconv3(output, output_size=(bs, 16, 8, 8))
        output = self.act3(output)
        output=self.activation(output)

        output_mu = self.deconv4(output, output_size=(bs, 3, 8, 8))
        #output_mu= self.activation(output_mu)

        output_logvar = self.deconv5(output, output_size=(bs, 3, 8, 8))
        output_logvar= -self.relu(output_logvar)
        return output_mu, output_logvar



class VAE_Generator(nn.Module):
    def __init__(self, input_channels, hidden_size, representation_size=(32, 2, 2)):
        super(VAE_Generator, self).__init__()
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.representation_size = representation_size
        
        self.encoder = Encoder(input_channels, hidden_size)
        self.decoder = Decoder(hidden_size, representation_size)
        
    def forward(self, x):
        batch_size = x.size()[0]
        mean, logvar = self.encoder(x)
        std = logvar.mul(0.5).exp_()
        
        for i in range (10):
            reparametrized_noise = Variable(torch.randn((batch_size, self.hidden_size))).cuda()

            reparametrized_noise = mean + std * reparametrized_noise
            if i==0:
                rec_images,var_image = self.decoder(reparametrized_noise)
            else:
                rec_images_tmp,var_image_tmp=self.decoder(reparametrized_noise)
                rec_images=torch.cat([rec_images,rec_images_tmp],0)
                var_image=torch.cat([var_image,var_image_tmp],0)
        return mean, logvar, rec_images,var_image



#################################