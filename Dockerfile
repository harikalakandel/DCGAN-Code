FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04


RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

#RUN apt-key adv --fetch-keys https://conda.binstar.org/menpo opencv


RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        python3.9 \
        python-dev \
        python3-pip \
        python-setuptools \
        && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update

#RUN pip install torch --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu116 #??? not sure but trying for cuda 11.6

#RUN pip install --upgrade pip && \
#    pip install --no-cache-dir numpy==1.21.0 scipy pandas scikit-learn numba tensorflow nilearn scikit-image keras zipfile37 matplotlib torch torchvision pytorch-lightning==1.5.8 nsdcode nibabel albumentations GPUtil

#downgrading https://candid.technology/runtimeerror-cudnn-error-cudnn-status-not-initialized/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir numpy==1.21.0 scipy pandas scikit-learn numba tensorflow nilearn scikit-image keras zipfile37 matplotlib torch==1.7.1 torchvision==0.8.2 pytorch-lightning==1.5.8 nsdcode nibabel albumentations GPUtil config utils psutil




RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python



#opencv-contrib-python d21

#  jupyter spyder torchvision accelerate fastprogress opencv-contrib-python pytorch-lightning>=1.4 torchmetrics>=0.6 d21


#RUN !pip install git+https://github.com/PyTorchLightning/pytorch-lightning

#RUN pip install --no-cache-dir --upgrade pip && \
#pip install --no-cache-dir numpy scipy pandas scikit-learn numba tensorflow nilearn scikit-image keras zipfile37 jupyter spyder matplotlib gputil torch torchvision accelerate     fastprogress

WORKDIR /App
