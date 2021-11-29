import math
import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
import torch.optim as optim

def call_bn(bn, x):
    return bn(x)

class data_model(nn.Module):
    def __init__(self, input_channel=512, n_outputs=8, dropout_rate=0.5, top_bn=False):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        super(data_model, self).__init__() 
        self.l_c1=nn.Linear(16*512,128)
        self.l_c2=nn.Linear(128,n_outputs)
        self.dropout_1=nn.Dropout(dropout_rate)
    
    def forward(self, x):
        h=x
        h = h.view(h.size(0), -1)
        h=self.l_c1(h)
        h=F.relu(h)
        #h=F.dropout(h, p=self.dropout_rate)
        h=self.dropout_1(h)
        h=self.l_c2(h)
        return h
    
    def reset_parameters(self):
        for i,m in enumerate(self.modules()):
            if(i>0):
                if not isinstance(m, nn.Dropout):
                    m.reset_parameters()