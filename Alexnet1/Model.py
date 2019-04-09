import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


use_cuda=torch.cuda.is_available()

class CNN(nn.Module): #always start with inherit nn.Module

    def __init__(self):
        super(CNN, self).__init__()
        conv1=nn.Conv2d(1,6,5,1)
        #activation ReLU
        pool1=nn.MaxPool2d(2)
        conv2=nn.Conv2d(6,16,5,1)
        #activation ReLU
        pool2=nn.MaxPool2d(2)

        self.conv_module=nn.Sequential(
            conv1,
            nn.ReLU(),
            pool1,
            conv2,
            nn.ReLU(),
            pool2
        )

        fc1=nn.Linear(16*4*4, 120)
        # activation ReLU
        fc2 = nn.Linear(120, 84)
        # activation ReLU
        fc3 = nn.Linear(84, 10)

        self.fc_module=nn.Sequential(
            fc1,
            nn.ReLU(),
            fc2,
            nn.ReLU(),
            fc3
        )

        #allocate gpu
        if use_cuda:
            self.conv_module=self.conv_module.cuda()
            self.fc_module=self.fc_module.cuda()

    def forward(self, x):
        out=self.conv_module(x)
        #make linear
        dim=1
        for d in out.size()[1:]:
            dim=dim*d
        out=out.view(-1,dim)
        out=self.fc_module(out)
        return F.softmax(out, dim=1)
