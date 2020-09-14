import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import models


class EmbeddingNet(nn.Module):
    def __init__(self,arch,pretrained=True,fci=51200):
        super(EmbeddingNet,self).__init__()
        self.inf = fci
        self.convnet = nn.Sequential(*list(arch.children())[:-2])
        
        self.fc = nn.Sequential(nn.Linear(self.inf,128),
                               nn.Sigmoid())
        
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
    
class TripletNet(nn.Module):
    
    def __init__(self,arch,pretrained=False,fci = 131072):
        
        super(TripletNet,self).__init__()
        
        self.pretrained = pretrained
        self.AnchorNet = EmbeddingNet(arch,self.pretrained,fci)
        self.PositiveNet = EmbeddingNet(arch,self.pretrained,fci)
        self.NegativeNet = EmbeddingNet(arch,self.pretrained,fci)
    
    def forward(self,x1,x2,x3):
#         print(x1.shape)
        output1 = self.AnchorNet(x1)
        output2 = self.PositiveNet(x2)
        output3 = self.NegativeNet(x3)
#         print(output1.shape)
        return (output1,output2,output3)
    
    def GetEmbeddings(self,x1):
        embeddings = self.AnchorNet(x1)
        return embeddings