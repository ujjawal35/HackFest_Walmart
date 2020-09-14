
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
import cv2
from tqdm import tqdm

# In[2]:

def get_list(df,fn,label):
    
    fns = list(df[fn])
    labels = (df[label])
    
#     print(fns)
    
    dic = dict(zip(fns,labels))
#     print(dic)
    cp = pd.MultiIndex.from_product([fns,fns,fns],names = ['anchor','positive','negative'])
    fnm = pd.DataFrame(index= cp).reset_index()
    
    ind = []
    
    for i in tqdm(range(fnm.shape[0])):
        
        if fnm.iloc[i][0]==fnm.iloc[i][1]:
            ind.append(i)
        elif dic[fnm.iloc[i][0]] != dic[fnm.iloc[i][1]]:
            ind.append(i)
        elif dic[fnm.iloc[i][0]] == dic[fnm.iloc[i][2]]:
            ind.append(i)
    df2 = fnm.drop(ind,axis=0)
    df2.reset_index(drop=True,inplace = True)
    
    return df2
    

def load_image(fn,size,path):
    img = cv2.imread(path+'/'+fn)
    img = cv2.resize(img,(size,size))
    img = F.to_tensor(img)
    
    return img


# In[ ]:


class MyDataset(Dataset):
    
    def __init__(self,df,path,fn = 'fn',label = 'label',size=300):
        
#         self.ids = list(df['image_name'])
#         self.labels = list(df['x1'])
        df = get_list(df,fn,label)
        self.size = size
        self.path = path
        self.anchor = list(df['anchor'])
        self.positive = list(df['positive'])
        self.negative = list(df['negative'])
        

        
    def __len__(self):
        return len(self.anchor)
    
    def __getitem__(self,index):

        x1 = load_image(str(self.anchor[index]),self.size,self.path)
        x2 = load_image(str(self.positive[index]),self.size,self.path)
        x3 = load_image(str(self.negative[index]),self.size,self.path)

        return (x1,x2,x3),[]
        
        
        
        

