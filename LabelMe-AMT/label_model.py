import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import sklearn.metrics as metrics
import sys

class NaiveBayes():
    def __init__(self,n_annot,class_num,prior):
        self.e_conf = np.zeros((n_annot,class_num,class_num))
        self.n_annot = n_annot
        self.class_num = class_num
        self.prior= prior
        
    def train_model(self,resp_org,one_hot_label):
        # est_conf
        # computes posterior probability distribution of the true label given the noisy labels annotated by the workers
        # and pseudo one_hot label 
        n = resp_org.shape[0]
        m = resp_org.shape[1]
        k = resp_org.shape[2]
        self.e_conf = np.zeros((m,k,k))
        temp_conf = np.zeros((m,k,k))
        
        #Estimating confusion matrices of each worker by assuming pseudo one-hot label is the ground truth label
        for i in range(n):
            workers_this_example=[]
            for x in range(m):
                if(resp_org[i,x,0]!=-1):
                    workers_this_example.append(x)
            for j in workers_this_example: #range(m)
                temp_conf[j,:,:] = temp_conf[j,:,:] + np.outer(one_hot_label[i],resp_org[i,j])
        #regularizing confusion matrices to avoid numerical issues
        #print(temp_conf)
        for j in range(m):  
            for r in range(k):
                if (np.sum(temp_conf[j,r,:]) ==0 or np.isnan(sum(temp_conf[j,r,:]))):
                    # assuming worker is spammer for the particular class if there is no estimation for that class for that worker
                    temp_conf[j,r,:] = 1.0/k
                else:
                    # assuming there is a non-zero probability of each worker assigning labels for all the classes
                    temp_conf[j,r,:][temp_conf[j,r,:]==0] = 1e-10
            self.e_conf[j,:,:] = np.divide(temp_conf[j,:,:],np.outer(np.sum(temp_conf[j,:,:],axis =1),np.ones(k)))
        return self.e_conf

    def infer(self, resp_org):
        n = resp_org.shape[0]
        m = resp_org.shape[1]
        k = resp_org.shape[2]
        e_class = np.zeros((n,k))
        temp_class = np.zeros((n,k))
        for i in range(n):
            workers_this_example=[]
            for x in range(m):
                if(resp_org[i,x,0]!=-1):
                    workers_this_example.append(x)
            for j in workers_this_example: 
                if (np.sum(resp_org[i,j]) ==1):
                    temp_class[i] = temp_class[i] + np.log(np.dot(self.e_conf[j,:,:],np.transpose(resp_org[i,j])))
            temp_class[i] = np.multiply(np.exp(temp_class[i]),self.prior)
            #temp_class[i] = np.exp(temp_class[i])
            temp_class[i] = np.divide(temp_class[i],np.outer(np.sum(temp_class[i]),np.ones(k)))
            e_class[i] = temp_class[i]           
        return e_class
    
    def evaluate(self,resp_org,true_label_numpy):
        outputs3=self.infer(resp_org)
        pred=np.argmax(outputs3,axis=1)
        acc=metrics.accuracy_score(true_label_numpy,pred)
        return acc