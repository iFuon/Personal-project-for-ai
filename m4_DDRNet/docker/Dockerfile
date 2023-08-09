ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda clean --all

# Install MMCV
ARG PYTORCH
ARG CUDA
ARG MMCV
RUN ["/bin/bash", "-c", "pip install --no-cache-dir mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${PYTORCH}/index.html"]

# # Install MMSegmentation
# RUN git clone https://github.com/open-mmlab/mmsegmentation.git /mmsegmentation
# WORKDIR /mmsegmentation
# ENV FORCE_CUDA="1"
# RUN pip install -r requirements.txt
# RUN pip install --no-cache-dir -e .

#Package requirements
RUN pip install easydict
RUN pip install opencv-python
RUN pip install Shapely
RUN pip install Cython
RUN pip install scipy
RUN pip install pandas
RUN pip install PyYAML
RUN pip install json-tricks
RUN pip install scikit-image
RUN pip install yacs
RUN pip install tensorboardX
RUN pip install ninja
#RUN pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

RUN git clone https://KangyueFu:ATBBs8mT7D5DbDzKa3CLEMZhjadV61C1FF49@bitbucket.org/kiriyachristin/comp3888_th16_03_christin.git /m4_DDRNet
WORKDIR /m4_DDRNet