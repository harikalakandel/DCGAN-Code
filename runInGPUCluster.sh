#!/bin/bash
#SBATCH --job-name=DCGAN_MRI_7T_MoreLayesDistMem
#SBATCH --output=DockerOut/DCGAN_MRI_7T_MoreLayersDistMem-stdout-%j.out
#SBATCH --error=DockerOut/DCGAN_MRI_7T_MoreLayersDistMem-err-%j.out
#SBATCH --ntasks=1
#SBATCH --mem=300G
#SBATCH --cpus-per-task=40
#SBATCH --gpus-per-node=4
#SBATCH --nodelist=slurmnode-gpu03
#SBATCH --partition=gpu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=h.kandel@bbk.ac.uk
srun --nodelist=slurmnode-gpu03 docker run --gpus all --rm -v /home/hkande01/testInGPU_Cluster:/App cuda-fmri:hari python3 scripts/runDCGANModel3DUsingMRI_7TWithMultiLayers_DistMemory.py 1>Outputs/output_DCGANModel3DUsingMsRI_7TWithMultiLayersDM.file 2>Outputs/error_DCGANModel3DUsingMRI_7TWithMultiLayersDM.file