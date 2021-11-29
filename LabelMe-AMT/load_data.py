 # -*- coding:utf-8 -*-
from __future__ import print_function
from data.labelme import LABELME
import torch
        
def load_data(DATASET, root='./data/', batch_size=128):
    print ('loading dataset...')
    train_dataset = LABELME(root=root, 
                                 data_type='train')
    val_dataset = LABELME(root=root, 
                               data_type='valid')
    test_dataset = LABELME(root=root, 
                               data_type='test')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               num_workers=1,
                                               drop_last=False,
                                               shuffle=True)
    train_loader_no_random = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               num_workers=1,
                                               drop_last=False,
                                               shuffle=False)    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=batch_size, 
                                              num_workers=1,
                                              drop_last=False,
                                              shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size*3, 
                                              num_workers=1,
                                              drop_last=False,
                                              shuffle=True)
    return train_loader, train_loader_no_random, val_loader, test_loader