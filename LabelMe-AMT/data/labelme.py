# -*- coding:utf-8 -*-

from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
import torchvision.transforms as transforms
import torch.utils.data
def load_data(filename):
    f = open(filename,'rb')
    data = np.load(f)
    f.close()
    return data
class LABELME(data.Dataset):
    def __init__(self, root, 
                 transform=None, target_transform=None,data_type='train', indices=range(10000)):
        DATA_PATH = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.data_type = data_type  # training set or test set
        self.dataset='labelme'
        self.nb_classes=8

        if(data_type=='train'):
             print ("\nLoading train data...")
             # images processed by VGG16
             data_train_vgg16 = load_data(DATA_PATH+"data_train_vgg16.npy")[indices]
             print (data_train_vgg16.shape)#(10000,4,4,512)

             # ground truth labels
             labels_train = load_data(DATA_PATH+"labels_train.npy")[indices]

             self.train_data=data_train_vgg16
            
             # labels obtained from majority voting
             labels_train_mv = load_data(DATA_PATH+"labels_train_mv.npy")[indices]
             print (labels_train_mv.shape)#(10000,)

             # data from Amazon Mechanical Turk
             print ("\nLoading AMT data...")
             answers = load_data(DATA_PATH+"answers.npy")[indices]
             print (answers.shape)#(10000,59)
             N_ANNOT = answers.shape[1]
             self.train_noisy_labels=answers
             if __name__=='__main__':
                 a=0
                 for i in range(1000):
                     for x in answers[1000:2000]:
                         a+=(x==answers[i]).all()
                 print(a)
             self.train_noisy_labels=np.concatenate((self.train_noisy_labels,np.expand_dims(labels_train_mv,axis=1)),1)
             self.train_noisy_labels=np.concatenate((self.train_noisy_labels,np.expand_dims(labels_train,axis=1)),1) 

        elif(data_type=='valid'):
             # load valid data
             print ("\nLoading valid data...")

             # images processed by VGG16
             data_valid_vgg16 = load_data(DATA_PATH+"data_valid_vgg16.npy")
             print (data_valid_vgg16.shape)#(500,4,4,512)

             # valid labels
             labels_valid = load_data(DATA_PATH+"labels_valid.npy")
             print (labels_valid.shape)#(500,)
             
             probe_data = np.zeros((80,4,4,512),np.float32)
             probe_label = np.zeros(80,np.float32)
             for i in range(self.nb_classes):
                 probe_data[10*i:10*(i+1)]=data_valid_vgg16[labels_valid==i][:10]
                 probe_label[10*i:10*(i+1)]=labels_valid[labels_valid==i][:10]
                 
             self.valid_data =probe_data
             self.valid_labels=probe_label
        else:
             # load test data
             print ("\nLoading test data...")

             # images processed by VGG16
             data_test_vgg16 = load_data(DATA_PATH+"data_test_vgg16.npy")
             print (data_test_vgg16.shape)#(1188,4,4,512)

             # test labels
             labels_test = load_data(DATA_PATH+"labels_test.npy")
             print (labels_test.shape)#(1188,)

             self.test_data =data_test_vgg16
             self.test_labels=labels_test
             
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.data_type=='train':
            img, target = self.train_data[index], self.train_noisy_labels[index]
        elif self.data_type=='valid':
            img, target = self.valid_data[index], self.valid_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        if self.data_type=='train':
            return len(self.train_data)
        elif self.data_type=='valid':
            return len(self.valid_data)        
        else:
            return len(self.test_data)