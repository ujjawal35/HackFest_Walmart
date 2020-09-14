import torch
import torch.nn as nn
import pandas as pd
from torchvision import models
import torchvision
import torch.nn.functional as F
from torchvision import*
import numpy as np

from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
import cv2

def load_image(fn,size,path,aug,y):
    img2 = cv2.imread(path+'/'+fn)
    img = cv2.resize(img2,(size,size))
#     print(img.shape)
    if aug == 0:
        yy = y
    if aug == 1:
    #rotate 90 clock
        img = cv2.transpose(img)
        img = cv2.flip(img,1)
        yy = [y[3],y[0],y[1],y[2]]
    elif aug == 2:
    #rotate 90 anticlock
        img = cv2.transpose(img)
        img = cv2.flip(img,1)
        yy = [y[1],y[3],y[2],y[0]]
    elif aug == 3:
    # flip v
        img = cv2.flip(img,0)
        yy = [y[0],y[3],y[2],y[1]]
    elif aug == 4:
    # flip h
        img = cv2.flip(img,1)
        yy = [y[2],y[1],y[0],y[3]]
        
    img = F.to_tensor(img)
    yy = torch.tensor(np.array(yy),dtype = torch.float)
    return img, yy


class MyDataset(Dataset):
    
    def __init__(self,df,path,size=300,aug = True):
        
#         self.ids = list(df['image_name'])
#         self.labels = list(df['x1'])
        self.size = size
        self.path = path
        self.aug = aug
        self.fn = list(df['image_name'])
        self.x1 = list(df['x1']/640*size)
        self.y1 = list(df['y1']/480*size)
        self.x2 = list(df['x2']/640*size)
        self.y2 = list(df['y2']/480*size)
        

        
    def __len__(self):
        return len(self.fn)
    
    def __getitem__(self,index):
        
        aaug = 0
#         if self.aug == True:
#             aaug = np.random.randint(0,1+1)
        y = [self.x1[index],self.y1[index],self.x2[index],self.y2[index]]
        x,y = load_image(str(self.fn[index]),self.size,self.path,aaug,y)
        
        return (x),y
    

class cnn_network(nn.Module):
    def __init__(self,arch,pretrained=True,fci=51200):
        super(cnn_network,self).__init__()
        self.inf = fci
        self.convnet = nn.Sequential(*list(arch.children())[:-2])
        
        self.fc = nn.Sequential(nn.Linear(self.inf,4))
#                                nn.ReLU())
        
    def forward(self,x):
        output = self.convnet(x)
#         self.inf = self.num_flat_features(output)
        output = output.view(output.size()[0],-1)
        output = self.fc(output)
        return output
        
    def num_flat_features(self,t):
#         set_trace()
        size = t.size()[1:]
        o = 1
        for s in size:
            o *= s
        return o
        