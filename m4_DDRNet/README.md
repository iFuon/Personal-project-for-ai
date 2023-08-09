# DDRNet (m4)

- Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes describes the use of efficient methods aiming to improve runtime while retaining semantic segmentation accuracy ([paper](https://paperswithcode.com/paper/deep-dual-resolution-networks-for-real-time)).  
- Our model repository is adapted from official git hub repository of [DDRNet](https://github.com/chenjun2hao/DDRNet.pytorch), and it presents two model variations ddrnet23 and ddrnet23_slim, with the latter producing slightly faster runtime, but lower mIOU. View DDRNet_model.html for a summary of the model environment and execution.
- The cityscapes data set is used.

## Structure  
- The file structures are unchanged from the online repository link, however some key files are of note.  
- All direct configuration files are in ```experiments/cityscapes/```.   
- Model program files are in ```lib/```, while its execution files are in ```tools/```.  
- Pre-trained model files to support the learning process are in ```pretrained_models/```  
- All config files need the pretrained model ```DDRNet23s_imagenet.pth``` Please download ImageNet from [official](https://github.com/ydhongHIT/DDRNet)  
- ````environment.yaml```, which specifies the runtime conditions.  
- The log files are in ```log/```  

## Instructions  
You need to train the model first before benchmarking it. The train.py file will create .pth files which is need for evaluation.   
Firstly, (connect to sc9) run ```git clone https://KiriyaChristin@bitbucket.org/kiriyachristin/comp3888_th16_03_christin.git```) and then ```cd``` to the project folder ```m4_DDRNet/```      
Then, Follow the steps below to build the Docker environment for training the model.
```diff
- Make sure you are always in the project folder
```
### Docker
Run command lines below one by one:  
1. build the docker environment for the model  

```
docker build -t ddr_m4 docker/ 
```
2. Once you run the command line above to build the environment, you only need to run docker run.  
   below for running the built docker environment of this model  
```
docker run --gpus all --shm-size=8g -it -v /home/usyd-05a/data:/m4_DDRNet/m4_DDRNet/data ddr_m4 
```
3. In the container, move into workpalce:
```
cd m4_DDRNet/
```

### The command line for training:  
```
python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --cfg experiments/cityscapes/ddrnet23_slim.yaml
```
Note: specify --local_rank -1 if you get error related to local rank when training.
#### Command line arguments avaliable:
<pre>
  -h, --help            show this help message and exit
  --cfg CFG             default ddrnet23_slim.yaml, experiment configure file name
  --seed SEED
  --local_rank          default -1, specify integer bigger than -1 for distributed training
  --benchmark           default 2, 1 for AMP, 2 for baseline, 3 for channel.last and 4 for channel.last and AMP
</pre>

For example, to switch on/off optimiztion method AMP, use the command line below.  
```
python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --cfg experiments/cityscapes/ddrnet23_slim.yaml --benchmark 1
```  
```diff
! However: You can only use local_rank = -1  
```
Please look through ```train.py``` and other python files in ```lib/``` for detailed coding of this model.  

### The command line for evaluation:  
```
python tools/eval.py --cfg experiments/cityscapes/ddrnet23_slim.yaml
```  
### After training
After finish training, please follow the step below to export the log files for visualizing scalars.  
1. Exit Docker and Commit image
```
exit
docker commit ${CONTAINER_ID} ddr_m4
```
2. Export the log file form Docker to current directory (project folder)
```
docker cp ${CONTAINER_ID}:m4_DDRNet/m4_DDRNet/log .
```  

or Export the log file form Docker to any path
```
docker cp ${CONTAINER_ID}:m4_DDRNet/m4_DDRNet/log ${the path you want to save the log files}
```

### Tensorboards for visualize scalars
1. Follow the [instruction](https://github.com/pytorch/kineto/blob/main/tb_plugin/README.md) to install the tensorboards  
2. Run command line
```
tensorboard --logdir = ./log/cityscapes/${the config file name you already used to train} --bind_all
```  
or run command line below if your log file is not in the current directory (project folder)  
```
tensorboard --logdir = ${the path to where you saved the log files}/log/cityscapes/${the config file name you already used to train} --bind_all
```
eg. run  
```
tensorboard --logdir=./log/cityscapes/ddrnet_23_slim --bind_all
```  
3. Open the website <u>http://${your server's IP address}:${port number}/</u> in the browser. e.g. access <u>http://192.168.10.9:6006/</u>

# Modification
The below modifications to config files can be made  
1. Modify END_EPOCH to change the epoch number of epoch  
2. Modify BATCH_SIZE_PER_GPU, and so the iteration number in each epoch will also change  
3. Modify LR to change learning rate   
4. Modify MOMENTUM  

# Improvement of DDRNet (m4)  
## Training time  
It takes around 7.5 second to finish training on 1 epoch (which reduced half time before improvement)  
## Config  
- The config file is ```ddrnet23_slim.yaml```  
### Paramaters  
- 185 iteration of every epoch and total epoch is 484  
- test set: val data in cityscapes, images used refer to 'data/list/cityscapes/val.lst'  
- train set: train data in cityscapes, images used refer to 'data/list/cityscapes/train.lst'  
- pretrained model: path refers to "pretrained_models/DDRNet23s_imagenet.pth"  
- BATCH_SIZE_PER_GPU: 8  
- OPTIMIZER: sgd  
- MULTI_SCALE: true  
- DOWNSAMPLERATE: 1  
- SCALE_FACTOR: 16
- FLIP_TEST: false
## Speedup 
We use CUDA AUTOMATIC MIXED PRECISION method (add autocast) to scale the loss  