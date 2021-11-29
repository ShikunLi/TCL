# -*- coding:utf-8 -*-
from __future__ import print_function
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def evaluate(test_loader, data_model):
    print ('Evaluating ...')
    data_model.eval()
    correct1=0
    total1=0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        outputs1 = data_model(images)
        _, pred1 = torch.max(outputs1, -1)
        outputs1 = F.softmax(outputs1,dim=-1)

        total1 += labels.shape[0]
        correct1 += (pred1==labels).sum()
        
    test_acc1=100*float(correct1)/float(total1)
    return test_acc1

    
def evaluate_label_model(test_loader,label_model,N_ANNOT,CLASS_NUM):
    print ('Evaluating ...')
    label_model.eval()
    correct=0
    total1=0
    for images, labels, _ in test_loader:
        labels = Variable(labels).cuda()
        true_labels=labels[:,-1]
        wokers_labels=labels[:,:-2]

        temp_org=np.zeros((wokers_labels.shape[0],N_ANNOT,CLASS_NUM))      
        for j,label in enumerate(wokers_labels.type(torch.LongTensor)):
            for i in range(N_ANNOT):
                if(label[i].item()!=-1):
                    temp_org[j,i,label[i].cpu().numpy()]=1
                else:
                    temp_org[j,i,:]=0# 没有就填0               
        outputs_3=label_model(torch.Tensor(temp_org).cuda())
        total1 += labels.shape[0]
        _, pred_label = torch.max(outputs_3, -1)
        correct+=(pred_label==true_labels).sum()
        
    eval_acc=100*float(correct)/float(total1)
    return eval_acc