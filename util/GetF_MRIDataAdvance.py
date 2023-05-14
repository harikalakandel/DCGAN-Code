# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 22:46:03 2022

@author: Harikala
"""

import os
import numpy as np
import glob
import nibabel as nib
import skimage.transform as skTrans


def load_fMRIDataFixSizeT2W1(x_range,y_range,z_range,workingFrom='PC'):
    
    myData = None
    myAnswer = None
    
    #elif workingFrom == 'UNI':
     #   dataPathX='NSD/nsddata/ppdata/subj0'+sId+'/func1mm/*.nii.gz'
    if workingFrom == 'PC':
        #folders = os.listdir('E:/fMRI-Dataset/RestingState-7T/R_fMRI1_Unprocessed')
        folders = ['164131', '164636','165436','167036']
    else:
        #folders = os.listdir('HCP/RestingState-7T/R_fMRI1_Unprocessed') 
        folders = ['164131', '164636','165436','167036']
        '''
        
        folders = ['100610','102311','102816','104416','105923','108323','109123','111312','111514','114823']
        
        folders = ['100610', '102311', '102816', '104416', '105923','108323', '109123', '111312', '111514', '114823',
                   '115017','115825','116726', '118225', '125525', '126426', '128935', '130114', '130518',
                   '131722', '132118','134627', '134829', '135124', '137128', '140117', '144226', '145834', '146129',
                   '146432', '146735', '146937', '148133','150423', '155938', '156334', '157336', '158035', '158136',
                   '159239', '162935', '164131', '164636','165436','167036']
        
        
        folders = ['100610', '102311', '102816', '104416', '105923','108323', '109123', '111312', '111514', '114823',
                   '115017','115825','116726', '118225', '125525', '126426', '128935', '130114', '130518', '131722',
                   '132118','134627', '134829', '135124', '137128', '140117', '144226', '145834', '146129', '146432',
                   '146735', '146937', '148133','150423', '155938', '156334', '157336', '158035', '158136', '159239',
                   '162935', '164131', '164636','165436','167036', '167440', '169040', '169343', '169444',   '169747', '171633',
                   '172130', '173334', '175237', '176542', '177140', '177645','177746', '178142','178243', '178647', '180533', 
                   '181232', '181636', '182436', '182739', '185442', '186949', '187345', '191033', '191336', '191841', '192439',
                   '192641', '193845', '195041', '196144', '197348', '198653', '199655', '200210', '200311', '200614', 
                   '201515', '203418', '204521', '205220', '209228', '212419', '214019', '214524', '221319', '233326', '239136',
                   '246133', '249947', '251833', '257845', '263436', '283543', '318637', '320826', '330324', '346137']
      '''
      
        print('Number of Samples ',len(folders))
    
    for cSubject in folders:
      if workingFrom == 'PC':
          currFilePath = 'E:/fMRI-Dataset/RestingState-7T/R_fMRI1_Unprocessed/'+cSubject+'/unprocessed/7T/rfMRI_REST1_PA/'+cSubject+'_7T_rfMRI_REST1_PA_SBRef.nii.gz'
          currFilePathA = 'E:/fMRI-Dataset/Structural-7T/'+cSubject+'/MNINonLinear/T2w_restore.1.60.nii.gz'
      else:
          currFilePath = 'HCP/RestingState-7T/R_fMRI1_Unprocessed/'+cSubject+'/unprocessed/7T/rfMRI_REST1_PA/'+cSubject+'_7T_rfMRI_REST1_PA_SBRef.nii.gz'
          currFilePathA = 'HCP/Structural-7T/'+cSubject+'/MNINonLinear/T2w_restore.1.60.nii.gz'
          
     
      test_load = nib.load(currFilePath).get_fdata()
      test_loadA = nib.load(currFilePathA).get_fdata()
      
      tmpData = skTrans.resize(test_load, (x_range,y_range,z_range), order=1, preserve_range=True)
      tmpDataA = skTrans.resize(test_loadA, (x_range,y_range,z_range), order=1, preserve_range=True)
        
      #add one dimension for channel .. first dimension [channel,sample,x,y,z]
           
      tmpData=np.expand_dims(tmpData,axis=0)
      tmpData=np.expand_dims(tmpData,axis=0)
      
      tmpDataA=np.expand_dims(tmpDataA,axis=0)
      tmpDataA=np.expand_dims(tmpDataA,axis=0)
      #combining all fMRI data from each subject
      
      if not(myData is None):
        myData=np.concatenate((myData,tmpData),axis=0)
        myAnswer=np.concatenate((myAnswer,tmpDataA),axis=0)
        
      else:
        myData = tmpData[:,:,:,:,:]
        myAnswer = tmpDataA[:,:,:,:,:]       
    
    return myData,myAnswer

