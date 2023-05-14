# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 23:41:51 2022

@author: Harikala
"""

#From--->https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/2.%20DCGAN/train.py
"""
Training of DCGAN network on MNIST dataset with Discriminator
and Generator imported from models.py
"""
import argparse
import torch     
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.distributed as dist
import sys
import torch.multiprocessing as mp


from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


workingFrom = 'PC'#'UNI'#'PC'#'Cuda'

x_RANGE=64
y_RANGE=64
z_RANGE=64

if workingFrom=='UNI':
    import os

    os.system("set 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:51200'")



    
    sys.path.insert(1, 'projects/gpuDCGANModelIn3D')
    sys.path.insert(1, 'util')
else:
    sys.path.insert(1, '../projects/gpuDCGANModelIn3D')
    sys.path.insert(1, '../util')
    

from gpuDCGANModel3DWithMultiLayers import gpuGenerator3D, gpuDiscriminator3D

from fMRIDataset import *
from GetF_MRIDataAdvance import *


#RUN_IN_TPU = True
RUN_IN_TPU = False


from memoryManagement import *


def cleanup(dist):
     dist.destroy_process_group()  
 
#source
#https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/mnist-training.ipynb#scrollTo=Hx4YVNHametU
myData,answer = load_fMRIDataFixSizeT2W1(x_RANGE,y_RANGE,z_RANGE,workingFrom)
print('Data Loaded')


dataset= fMRIDataset(myData, answer)
#Source-->https://github.com/pytorch/pytorch/issues/47587
def main_fun(rank, world_size, args):
    # setup the process groups    
    print('Current Rank is ',rank)    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    args.rank = rank
    args.world_size = world_size
    args.gpu = rank
    args.distributed = True

    
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)#
    dist.barrier()#should be below

    
    
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    
    #model = Your_Model().to(rank)
    gen = gpuGenerator3D(0).to(args.rank)
    
    disc = gpuDiscriminator3D(0).to(args.rank)
    
    #optimizer = Your_Optimizer()
    
    opt_gen = optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    
    
    writer = SummaryWriter(f"logs/model")

    writer_real = SummaryWriter(f"logs/real")
    writer_fake = SummaryWriter(f"logs/fake")
    step = 0
    
    for epoch in range(args.epochs):
            
        for batch_idx, (fMRI,mri) in enumerate(dataloader):
            print('Initial Memory Status:::::::::::')
            display_gpu_info(args.rank)
            
            mri = mri.to(args.rank)
            
            
            fMRI=fMRI.to(args.rank)
                       
            fake = gen(fMRI)
            
            #print('##########################################################################')

            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            #print('Shape of mri',mri.shape)
            disc_real = disc(mri).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            output = disc(fake)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
            
            
            print('Device :',args.rank,' Iteration:', epoch,'Batch IDx',batch_idx, 'GPU usage:', format_bytes(torch.cuda.memory_allocated(args.rank)))
            
            
         
            # Print losses occasionally and print to tensorboard
            if args.rank == 0:
                print(
                    f"Epoch [{epoch}/{args.epochs}] Batch {batch_idx}/{len(dataloader)} \
                      Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
                )
                
                with torch.no_grad():
                    
                                
                    
                    
                    fake = gen(mri)
                    
                  
                    img_grid_real = torchvision.utils.make_grid(
                        #real[:32], normalize=True
                        mri[0,0,:,:,int(z_RANGE/2)],normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[0,0,:,:,int(z_RANGE/2)], normalize=True
                    )
                    

                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                    
                    writer.add_scalar("gen_loss", loss_gen, global_step=step)
                    writer.add_scalar("dis_loss", loss_disc, global_step=step)
                    
                    

                step += 1
            del mri, fMRI
    del gen
    del disc
    
            
            
            
        
                
    print('Training Completed...............')
    #cleanup(dist)
    
    
    writer_real.close()
    writer_fake.close()
    writer.close()
    cleanup(dist)
    free_gpu_cache(args.device)
    display_gpu_info(args.device)
    
    





if __name__ == '__main__':
    
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0002)
   

    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str, default="/home/wz/data_set/flower_data/flower_photos")

   
    
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
   
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    opt = parser.parse_args()

    mp.spawn(main_fun, 
             args=(opt.world_size, opt), 
             nprocs=opt.world_size,
             join=True)
    print("Completed::::::::::::::::::::::::")
    



