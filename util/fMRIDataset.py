# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 15:01:53 2022

@author: Harikala
"""
import torch

class fMRIDataset():
    def __init__(self, a,b):        
        self.answer=b
        self.myData = a
                
    def __len__(self):        
        return len(self.myData)
    
    
       
    
    def __getitem__(self, idx):      
        fMRI = torch.from_numpy(self.myData[idx]).type(torch.float32)
        mri = torch.from_numpy(self.answer[idx]).type(torch.float32)        
        return fMRI, mri