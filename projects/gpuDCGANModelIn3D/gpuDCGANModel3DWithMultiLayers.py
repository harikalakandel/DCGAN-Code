#From--->https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""
Discriminator and Generator implementation from DCGAN paper
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.00, 0.002)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.002)
        nn.init.constant_(m.bias.data, 0)    



class gpuGenerator3D(nn.Module):
    def __init__(self, ngpu):
       
        super(gpuGenerator3D, self).__init__()
        self.ngpu=ngpu
        #gen_output= gen_input * 2
       # in_channels=1
        #self.netG = nn.Sequential(
            
            
           
        self.ct1=nn.ConvTranspose3d( in_channels=1, out_channels=2, kernel_size=4, stride=2,padding= 1, bias=False)
        self.batchNorm1  = nn.BatchNorm3d(2 )
        self.relu1= nn.ReLU(True)
            
            # state size. (ngf*2) x 16 x 16
        self.ct2=nn.ConvTranspose3d( in_channels=2, out_channels=4, kernel_size=4, stride=2,padding= 1, bias=False)
        self.batchNorm2=nn.BatchNorm3d(4)
        self.relu2= nn.ReLU(True)
          
        '''
          # state size. (ngf) x 32 x 32            
        self.ct3=nn.ConvTranspose3d( in_channels=4, out_channels=8, kernel_size=4, stride=2,padding= 1, bias=False)
        self.batchNorm3=nn.BatchNorm3d(8)
        self.relu3=nn.ReLU(True)
      
     '''   
            
        self.ct4=nn.ConvTranspose3d( in_channels=4, out_channels=1, kernel_size=4, stride=2,padding= 1, bias=False)
        self.tanh=nn.Tanh()
        
        #)
            # state size. (nc) x 64 x 64
        

   
    def forward(self, input):
        #pass the input through our first set of CONVT => RELU => BN
        #layers
         input= self.ct1(input)
         input= self.relu1(input)
         input= self.batchNorm1(input)
        #pass the output from previous layer through our second
        #CONVT => RELU => BN layer set
         input= self.ct2(input)
         input= self.relu2(input)
         input= self.batchNorm2(input)
         #pass the output from previous layer through our last set
         #of CONVT => RELU => BN layers
         '''
         input= self.ct3(input)
         input= self.relu3(input)
         input= self.batchNorm3(input)
         '''
        
         #pass the output from previous layer through CONVT2D => TANH
         #layers to get our output
         input= self.ct4(input)
         output = self.tanh(input)
         #return the output
         return output
        
    

class gpuDiscriminator3D(nn.Module):
    def __init__(self, ngpu,alpha=0.2):
        super(gpuDiscriminator3D, self).__init__()
        self.ngpu=ngpu
        #self.netD = nn.Sequential(
            # input: N x channels_img x 64 x 64
            
         
        self.conv1=nn.Conv3d(in_channels=1, out_channels=2 , kernel_size=4, stride=2, padding=1, bias=False)
        self.leakyRelu1=nn.LeakyReLU(alpha, inplace=True)
            # state size. (ndf*4) x 8 x 8
            
           
        self.conv2=nn.Conv3d(in_channels=2, out_channels=4 , kernel_size=4, stride=2, padding=1, bias=False)
        self.batchNorm2=nn.BatchNorm3d(4)
        self.leakyRelu2=nn.LeakyReLU(0.2, inplace=True)
            # state size. (ndf*8) x 4 x 4
        '''
        self.conv3=nn.Conv3d(in_channels=4, out_channels=8 , kernel_size=4, stride=2, padding=1, bias=False)
        self.batchNorm3=nn.BatchNorm3d(8)
        self.leakyRelu3=nn.LeakyReLU(alpha, inplace=True)
            # state size. (ndf*4) x 8 x 8
       ''' 
           
        self.conv4=nn.Conv3d(in_channels=4,out_channels= 1,  kernel_size=4, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, input):
        #pass the input through first set of CONV => RELU layers
        input= self.conv1(input)
        input= self.leakyRelu1(input)
        #pass the output from the previous layer through our second set of CONV => RELU layers 
        input= self.conv2(input)
        input=self.batchNorm2(input)
        input= self.leakyRelu2(input)
        #flatten the output from the previous layer and pass it
        #through our first (and only) set of FC => RELU layers
        '''
        input= self.conv3(input)
        input=self.batchNorm3(input)
        input= self.leakyRelu3(input)
        '''
        #pass the output from the previous layer through our sigmoid
        #layer outputting a single value
        input= self.conv4(input)
        output = self.sigmoid(input)
        #return the output
        return output