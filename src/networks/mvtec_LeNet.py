import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class MVTec_LeNet(BaseNet):

    def __init__(self):
        super().__init__()
        self.normalObj=None
            
        self.rep_dim = 128

        self.encoder=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=5),
            nn.ReLU(True),
            nn.Conv2d(64, 192, kernel_size=5, stride=3),
            nn.ReLU(True),
            nn.Conv2d(192, 384, kernel_size=5, stride=1),
            nn.ReLU(True),
            nn.Conv2d(384, 256, kernel_size=3, stride=3),
            nn.ReLU(True),
            nn.Conv2d(256, 50, kernel_size=4, stride=1),
            nn.ReLU(True))

        self.fc1 = nn.Linear(50, self.rep_dim, bias=False)


    def forward(self, x):
      x=self.normalObj.normalize(x)  
        
      x=self.encoder(x)
      x = x.view(x.size(0), -1)
      x=self.fc1(x)
      x=nn.ReLU(True)(x)   
         
      return x


class MVTec_LeNet_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.normalObj=None
        
        self.rep_dim = 128

        self.encoder=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=5),
            nn.ReLU(True),
            nn.Conv2d(64, 192, kernel_size=5, stride=3),
            nn.ReLU(True),
            nn.Conv2d(192, 384, kernel_size=5, stride=1),
            nn.ReLU(True),
            nn.Conv2d(384, 256, kernel_size=3, stride=3),
            nn.ReLU(True),
            nn.Conv2d(256, 50, kernel_size=4, stride=1),
            nn.ReLU(True))

        self.fc1 = nn.Linear(50, self.rep_dim, bias=False)
        self.fc2 = nn.Linear(self.rep_dim, 50, bias=False)
        
        self.decoder=nn.Sequential(
            
            nn.ConvTranspose2d(50, 256, kernel_size=4, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 384, kernel_size=3, stride=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(384, 192, kernel_size=5, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(192, 64, kernel_size=5, stride=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=11, stride=5),
            nn.ReLU(True))
            
    def forward(self, x):
        
        x=self.normalObj.normalize(x)
        
        
        x=self.encoder(x)
        
        # print("\nx0\n " , x.shape , "\n\n")
        x = x.view(x.size(0), -1)
        x=self.fc1(x)
        x=nn.ReLU(True)(x)
        x=self.fc2(x)
        x=nn.ReLU(True)(x)
        x = x.view(x.size(0), 50, 1, 1)
        


        x=self.decoder(x)
        
        return x