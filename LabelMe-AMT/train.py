# -*- coding:utf-8 -*-
from __future__ import print_function
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision.transforms as transforms
import argparse,sys
import datetime
import shutil
from data.labelme import LABELME
import math
from load_data import *
from global_info import *
from evaluate import *
from model_labelme import *
from label_model import *
from sklearn.isotonic import IsotonicRegression

parser=argparse.ArgumentParser()
parser.add_argument('--lr',type=float,default=0.0001)
parser.add_argument('--batch_size',type=int,default=128)
parser.add_argument('--pre_train_data_epoch',type=int,default=2)
parser.add_argument('--each_round_data_epoch',type=int,default=1)
parser.add_argument('--max_round',type=int,default=30)
parser.add_argument('--retrain_data_epoch',type=int,default=20)
parser.add_argument('--finetuning_epoch',type=int,default=10)
parser.add_argument('--labels_model_type', type=str, default='NB')
parser.add_argument('--dataset', type=str, default='labelme')
#其他参数
parser.add_argument('--cuda_dev', type=int, default=0)
parser.add_argument('--num_of_traning_times', type=int, default=1)
parser.add_argument('--input_address', type=str, default='./data/prepared/')
parser.add_argument('--output_address', type=str, default='./output/')
parser.add_argument('--experiment_name', type=str, default='labelme')
args=parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=str(args.cuda_dev)
#---------------------------------------------------------------------------------------------------------
if not os.path.isdir(args.output_address):
    os.makedirs(args.output_address)
__console__=sys.stdout
log_file=open(args.output_address+args.experiment_name+".log",'a')
sys.stdout=log_file
#---------------------------------------------------------------------------------------------------------
dataset_info.set_dataset(DATASET=args.dataset)
CLASS_NUM, N_ANNOT, N_FEATURE, N_TRAIN, N_VAL, N_TEST, prior=dataset_info.get_info()
#---------------------------------------------------------------------------------------------------------
pseudo_labels=np.zeros((N_TRAIN,CLASS_NUM))
workers_labels_org=np.zeros((N_TRAIN,N_ANNOT,CLASS_NUM))
true_label_numpy=np.zeros((N_TRAIN))
#------------------------------------------------------------------------------------------------------
def main():
    train_loader, train_loader_no_random, val_loader, test_loader=load_data(args.dataset, root=args.input_address, batch_size=args.batch_size)
    init_labels(train_loader_no_random,val_loader)
    data_model_1,optimizer_1=build_data_model()
    label_model_NB=build_label_model()
    pre_train_label_model(label_model_NB)
    pre_train_data_model(train_loader,test_loader, data_model_1,optimizer_1)
    calibrate_confidence(val_loader,data_model_1)
    update_pseudo_labels(-1,train_loader_no_random,data_model_1,label_model_NB)
    for current_round in range(args.max_round):
        train_label_model_one_round(current_round,label_model_NB)
        train_data_model_one_round(current_round,train_loader,test_loader,data_model_1,optimizer_1)
        calibrate_confidence(val_loader,data_model_1)
        update_pseudo_labels(current_round,train_loader_no_random,data_model_1,label_model_NB)
    retrain_data_model(train_loader,val_loader,test_loader)
    
def one_hot(target, n_classes):
    targets = np.array([target]).reshape(-1)
    one_hot_targets = np.eye(n_classes)[targets]
    return one_hot_targets

def init_labels(train_loader_no_random,val_loader):
    print('init_labels')
    global workers_labels_org,pseudo_labels,true_label_numpy
    for images, labels,indexs in train_loader_no_random:
        labels = labels.type(torch.cuda.LongTensor)  
        true_labels=labels[:,-1]
        wokers_labels=labels[:,:-2]
        temp_org=np.zeros((labels.shape[0],N_ANNOT,CLASS_NUM))      
        for j,label in enumerate(wokers_labels.type(torch.LongTensor)):
            for i in range(N_ANNOT):
                if(label[i].item()!=-1):
                    temp_org[j,i,label[i].cpu().numpy()]=1
                else:
                    temp_org[j,i,:]=-1
        #workers_labels
        workers_labels_org[indexs]=temp_org
        #pseudo_labels
        temp_org[temp_org==-1]=0
        temp_soft_labels=np.sum(temp_org,axis=1)
        temp_soft_labels=temp_soft_labels/np.sum(temp_soft_labels,axis=1).reshape(-1,1)        
        pseudo_labels[indexs]=temp_soft_labels
        #true_label_numpy
        true_label_numpy[indexs]=true_labels.cpu().squeeze().detach().numpy()

def build_data_model():
    print ('building data model...')
    data_model_1=data_model()
    data_model_1.cuda()
    print (data_model_1.parameters)
    global optimizer_1
    optimizer_1= torch.optim.Adam(data_model_1.parameters(),lr=args.lr)

    return data_model_1,optimizer_1
    
def build_label_model():
    print ('building label model...')
    label_model_NB=NaiveBayes(n_annot=N_ANNOT,class_num=CLASS_NUM,prior=prior)
    return label_model_NB 

def pre_train_label_model(label_model_NB):
    print('pre_train_label_model')
    label_model_NB.train_model(workers_labels_org,pseudo_labels)
    acc=label_model_NB.evaluate(workers_labels_org,true_label_numpy)
    print('Pre_train NB trainset Acc %.4f %%'%acc)
    return label_model_NB

def pre_train_data_model(train_loader,test_loader, data_model_1,optimizer_1):
    print('pre_train_data_model')
    data_model_1.train()
    for epoch in range(args.pre_train_data_epoch): 
        train_data_model_one_epoch(train_loader,data_model_1,optimizer_1)
        test_data=evaluate(test_loader,data_model_1)
        print('Pretrain data Epoch [%d/%d] Test Accuracy on the %s test images: test_data%.4f %%' \
          % (epoch+1, args.pre_train_data_epoch, N_TEST, test_data))

def train_label_model_one_round(current_round,label_model_NB):
    print('train_label_model_one_round')
    label_model_NB.train_model(workers_labels_org,pseudo_labels)
    acc=label_model_NB.evaluate(workers_labels_org,true_label_numpy)
    print('Train round %d NB trainset Acc %.4f %%'%(current_round+1,acc))
    return label_model_NB

def train_data_model_one_round(current_round,train_loader,test_loader, data_model_1,optimizer_1):
    print('train_data_model_one_round')
    data_model_1.train() 
    for epoch in range(args.each_round_data_epoch): 
        train_data_model_one_epoch(train_loader,data_model_1,optimizer_1)
        test_data=evaluate(test_loader,data_model_1)
        print('Train round %d data Epoch [%d/%d] Test Accuracy on the %s test images: test_data%.4f %%' \
          % (current_round+1,epoch+1, args.each_round_data_epoch, N_TEST, test_data))

def retrain_data_model(train_loader,val_loader,test_loader):
    print('retrain_data_model')
    data_model_1,optimizer_1=build_data_model()
    for epoch in range(args.retrain_data_epoch): 
        train_data_model_one_epoch(train_loader,data_model_1,optimizer_1)
        test_data=evaluate(test_loader,data_model_1)
        print('Retrain data Epoch [%d/%d] Test Accuracy on the %s test images: test_data%.4f %%' \
          % (epoch+1, args.retrain_data_epoch, N_TEST, test_data))
    
    for epoch in range(args.finetuning_epoch): 
        finetuning(val_loader,data_model_1,optimizer_1)
        test_data=evaluate(test_loader,data_model_1)
        print('Retrain data Epoch [%d/%d] Test Accuracy on the %s test images: test_data%.4f %%' \
          % (epoch+1, args.finetuning_epoch, N_TEST, test_data)) 

def finetuning(val_loader,data_model_1,optimizer_1):
    data_model_1.train()
    for iteration, (images, labels,indexs) in enumerate(val_loader):
        images = Variable(images).cuda()
        labels = Variable(labels).type(torch.cuda.LongTensor)

        outputs1=data_model_1(images)
        
        loss_1 = F.cross_entropy(outputs1, labels)
        
        optimizer_1.zero_grad()
        loss_1.backward()
        optimizer_1.step()




def train_data_model_one_epoch(train_loader,data_model_1,optimizer_1):
    data_model_1.train()
    for iteration, (images, labels,indexs) in enumerate(train_loader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        pseudo_target=torch.Tensor(pseudo_labels[indexs]).cuda()
        true_labels=labels[:,-1]

        outputs1=data_model_1(images)
        
        loss_1 = F.kl_div(F.log_softmax(outputs1,-1),pseudo_target) 
        
        optimizer_1.zero_grad()
        loss_1.backward()
        optimizer_1.step()



def calibrate_confidence(val_loader,data_model_1):
    data_model_1.eval() 
    predict_val_data=np.zeros((N_VAL,CLASS_NUM))
    true_labels_val=np.zeros((N_VAL,CLASS_NUM))
    for iteration, (images, labels,indexs) in enumerate(val_loader):
        images=Variable(images).cuda()
        labels=Variable(labels).type(torch.cuda.LongTensor)
        
        outputs1 = data_model_1(images)
        pro=F.softmax(outputs1,dim=-1)
        predict_val_data[indexs]=pro.cpu().detach().numpy()
        true_labels_val[indexs,labels.cpu().squeeze().detach().numpy()]=1
    global iso_reg_data
    iso_reg_data=[]
    for i in range(CLASS_NUM):
        iso_reg_data.append(IsotonicRegression(y_min=0, y_max=1,out_of_bounds='clip').fit(predict_val_data[:,i],true_labels_val[:,i]))
    return iso_reg_data   
    
def update_pseudo_labels(current_round,train_loader_no_random,data_model_1,label_model_NB):
    print('update_pseudo_labels')
    global pseudo_labels
    label_total=0
    label_correct_1=0
    label_correct_2=0
    data_model_1.eval()
    for i,(images,labels,indexs) in enumerate(train_loader_no_random):
        images=Variable(images).cuda()
        labels=Variable(labels).type(torch.cuda.LongTensor)
    
        outputs1 = data_model_1(images)

        outputs_label=torch.Tensor(label_model_NB.infer(workers_labels_org[indexs])).cuda()
        _, pred_label = torch.max(outputs_label, -1)
        
        pro=F.softmax(outputs1,dim=-1)
        outputs_data=pro
        
        co_metric_1=get_co_metric(outputs_data,outputs_label)
        _, pred_before = torch.max(co_metric_1, -1)
  
        pro_after_iso_reg=get_pro_after_iso_reg(pro)
        outputs_data=pro_after_iso_reg
        co_metric_2=get_co_metric(outputs_data,outputs_label)
        _, pred_after = torch.max(co_metric_2, -1)

        pseudo_labels[indexs]=co_metric_2.cpu().detach().numpy()
        
        label_total+=labels.shape[0]
        true_labels=labels[:,-1]
        
        label_correct_1+=(pred_before==true_labels).sum()
        label_correct_2+=(pred_after==true_labels).sum()
    pse_acc_1=(float(100)*float(label_correct_1)/float(label_total))
    pse_acc_2=(float(100)*float(label_correct_2)/float(label_total))
    print('Train round %d update_pseudo_labels Train before Acc %.4f %%'%(current_round+1,pse_acc_1))
    print('Train round %d update_pseudo_labels Train after Acc %.4f %%'%(current_round+1,pse_acc_2))
    return pse_acc_1,pse_acc_2

def get_pro_after_iso_reg(pro):
    np.set_printoptions(threshold=np.inf)
    torch.set_printoptions(profile="full")
    global iso_reg_data,temp_pro
    temp_pro=pro.cpu().detach().numpy()
    for i in range(CLASS_NUM):
        temp_pro[:,i]=iso_reg_data[i].predict(temp_pro[:,i])
    pro_after_iso_reg=normalize(torch.Tensor(temp_pro+1e-20).cuda())
    return pro_after_iso_reg
    
def get_co_metric(outputs_data,outputs_label):
    co_metric=normalize(outputs_data*outputs_label/torch.Tensor(prior).cuda())
    return co_metric
    
def normalize(v):
    return v/torch.sum(v,-1).view(-1,1)
    
    
    
if __name__=='__main__':  
    for i in range(args.num_of_traning_times):
        print('training time No.'+str(i+1))
        main()
    log_file.close()
    sys.stdout=__console__
