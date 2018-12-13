

from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import logging
import urllib.request
import zipfile
import glob
import datetime
import pickle
import argparse
from collections import Counter
import gc
import torch
from datetime import datetime
from datetime import date
import math
from ast import literal_eval
import random


class MyDataset(Dataset):
    """CMS data dataset."""

    def __init__(self, csv_file, root_dir, start_dt, end_dt, len_vec, keep_feature_dim, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.target_label = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.len_vec = len_vec
        self.keep_feature_dim = keep_feature_dim
        self.transform = transform

    def __len__(self):
        return len(self.target_label)

    def __getitem__(self, idx):
        data_name = os.path.join(self.root_dir,
                                self.target_label.iloc[idx, 0])
        
        data = pd.read_csv(data_name,index_col=None, header=0)
        data.columns = ['CLM_FROM_DT','VEC_LIST','Len2Pad']
        data['VEC_LIST'] = data['VEC_LIST'].apply(literal_eval)
        data['CLM_FROM_DT'] = data['CLM_FROM_DT'].apply(lambda x: pd.to_datetime(str(x), format='%Y-%m-%d'))
        temp_df=pd.DataFrame()
        temp_df['CLM_FROM_DT'] = pd.date_range(start_dt, end_dt, freq='D')
        temp_df = pd.merge(temp_df,data,on='CLM_FROM_DT',how='left')
        temp_df = temp_df.iloc[0:365]
        temp_df['VEC_LIST'] = temp_df['VEC_LIST'].apply(lambda x: [0.0]*len_vec*keep_feature_dim if isinstance(x,float) else x)
        output = np.array(temp_df['VEC_LIST'].values.tolist())
        
        label = self.target_label.iloc[idx, 1]
        label = np.array([label])
        sample = {'data': output, 'labels': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data, label = sample['data'], sample['labels']
        
        return {'data': torch.from_numpy(data),
                'labels': torch.from_numpy(label)}
        
         
save_path = r'/cms_data/train_s/'  
start_dt = '2008-1-1'
end_dt = '2008-12-31'
len_vec = 200
keep_feature_dim = 15

dataset = MyDataset(csv_file=r'/cms_data/train_s/sample_label.csv',
                                    root_dir=save_path,
                                    start_dt=start_dt,
                                    end_dt=end_dt,
                                    len_vec=len_vec,
                                    keep_feature_dim=keep_feature_dim,
                                    transform=ToTensor()
                                    )


  
    
# Use BCE 
            
            # added BN, change the output to 1d
class Claim_CNN(nn.Module):
    
#    def __init__(self, args):
    def __init__(self):
        super(Claim_CNN,self).__init__()
        
#        self.args = args        
#        out_c = args.out_c
#        len_vec = args.len_vec
#        ngram = args.ngram
#        ngram_w_dilation = args.ngram_w_dilation 
#        dilation_w = args.dilation_w
#        dropout = args.dropout
        
        out_c = 80
        len_vec = 200
        ngram = [2,3,4,5]
        ngram_w_dilation = [2,3,4]
        dilation_w = [2,3]
        dropout = 0.5
        
        self.conv1 = nn.Conv2d(1, out_c, (1, len_vec), stride=(1,len_vec)).double()
        self.convs1 = nn.ModuleList([
                nn.Conv2d(out_c, out_c, (1,window_size))
                for window_size in ngram
                ]).double()    
        self.convs2 = nn.ModuleList([
                nn.Conv2d(out_c, out_c, (1,window_size), dilation = (1, dila_w))
                for window_size in ngram_w_dilation for dila_w in dilation_w
                ]).double()
    
        self.conv4 = nn.Conv2d(out_c,out_c,(4,1),dilation=(7,1)).double()
    
        self.dropout = nn.Dropout(dropout)
        self.conv2_bn = nn.BatchNorm2d(out_c).double()
        
        self.fc1 = nn.Linear(320*out_c, 80*out_c).double()
        self.fc2 = nn.Linear(80*out_c, 1).double()
        self.fc = nn.Linear(320*out_c, 1).double()
        self.m = nn.Sigmoid()
        
        
        
    def forward(self,x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        xs = []
        for conv in self.convs1:
            x2 = F.relu(conv(x))        # [B, F, T, 1]
            x2 = F.max_pool2d(x2, (1,x2.size(3)))  # [B, F, 1]
            xs.append(x2)
            
        for conv in self.convs2:
            x3 = F.relu(conv(x))        # [B, F, T, 1]
            x3 = F.max_pool2d(x3, (1,x3.size(3)))  # [B, F, 1]
            xs.append(x3)
        
        x = torch.cat(xs,3)
        x = self.conv2_bn(x) 
        
        x5 = F.max_pool2d(x, (7,1),padding=(3,0))
        x5 = F.relu(self.conv4(x5))
        x5 = F.max_pool2d(x5,(2,1))       
        x5 = x5.view(x5.size(0), x5.size(1)*x5.size(2)*x5.size(3))
#        print(x5.size())
        
                      
        x6 = F.max_pool2d(x, (30,1),padding=(3,0))
        x6 = x6.view(x6.size(0), x6.size(1)*x6.size(2)*x6.size(3))
#        print(x6.size())
                
        x7 = F.max_pool2d(x, (91,1))
        x7 = x7.view(x7.size(0), x7.size(1)*x7.size(2)*x7.size(3))
#        print(x7.size())
        
        # Fully Connected Layer
        x = torch.cat((x5,x6,x7),1)        
#        print(x.size())
        x = self.dropout(x)  # (N, len(Ks)*Co)

        
        # output
        logits = self.m(self.fc((x)))
        
#        print(classes.size())
        
        return logits

    
cnn = Claim_CNN()
cnn.cuda()

EPOCH = 50        
BATCH_SIZE = 16
LR = 0.0005              # learning rate
#num_workers = 4


train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers = 8,  shuffle=True)

len(train_loader)
#torch.cuda.current_device() 
#torch.cuda.get_device_name(0) 
#torch.cuda.is_available()


optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.BCELoss()                       # the target label is not one-hotted

for epoch in range(EPOCH):
    print('EPOCH: ',epoch)
    for i_batch, s_b in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
#        print(i_batch, s_b['data'].size(),s_b['label'].size())
        b_x = s_b['data'].double().cuda()
        b_y = s_b['labels'].squeeze().double().cuda()    
        
        
              
        output1 = cnn(b_x)           # cnn output
        
#        print(output2.data.numpy())
        
        loss = loss_func(output1.squeeze().float(), b_y.float())   # cross entropy loss
        
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        
        if i_batch % 200 == 0:
            print('Loss: ',loss.cpu().data.numpy())
            print('Predicted: ',output1.cpu().squeeze().float().data.numpy())
            print('Actual: ',b_y.cpu().data.numpy())
            
torch.save(cnn.state_dict(), '/home/jing/Downloads/cms_data/train_s/CNN_TEST_MODELDICT.pt')
torch.save(cnn, '/home/jing/Downloads/cms_data/train_s/CNN_TEST_MODEL.pt')

