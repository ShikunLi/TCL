import numpy as np
class DatasetInfo():
    def __init__(self):
        pass    
    def set_dataset(self,DATASET='labelme'):
        if(DATASET=='labelme'):
            self.CLASS_NUM=8
            self.N_ANNOT=59
            self.N_FEATURE=16*512
            self.N_TRAIN=10000
            self.N_VAL=80 #500
            self.N_TEST=1188
            self.prior=np.ones(8)*1.0/8
    def get_info(self):
        return self.CLASS_NUM, self.N_ANNOT, self.N_FEATURE, self.N_TRAIN, self.N_VAL, self.N_TEST, self.prior

dataset_info=DatasetInfo()