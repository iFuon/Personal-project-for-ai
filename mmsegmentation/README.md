# Segformer in MMSegmentation repo (m3)

Part of the OpenMMLab project, MMSegmentation (mmseg) is an open source semantic segmentation toolbox that supports many other models by providing a consistent and unified runtime environment. (https://github.com/open-mmlab/mmsegmentation)

We are using it to run SegFormer, which describes the combination of a transformer with multilayer perceptron networks to improve performance and efficiency. (https://paperswithcode.com/paper/segformer-simple-and-efficient-design-for) 

## Structure
The file structures are unchanged from the online repository link, however some key files are of note.  
Our model files adapted from https://github.com/NVlabs/SegFormer  
- The configuration files for SegFormer are located in [configs/segformer]. It contains a .yml describing the parameters for different pre-trained model files that can be used to support the model. For example, segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py loads the mit-b0 file from OpenMMLabs for use on the cityscapes dataset.  
- [demo] contains demonstration files provided by the developer to understand the MMSegmentation toolkit.
- [docker] provides the necessary setup to run an mmseg Docker container. In particular [docker/serve] acts as the entrypoint.  
- [mmseg] stores the core program code for running the system.  
- [requirements] describe the necessary packages.  
- [tests] includes the training and testing scripts to run models and output results.  
- [tools] includes additional helper scripts such as dataset conversions, benchmarking, etc.  
- The README.md is the default readme file provided by the developers.  

## Instructions
Path: comp3888_th16_03_christin/mmsegmentation  
Build the image via Dockerfile, training in the image.  
Exit the image then commit so that you can export the logfile for training.  
Using tensorBoard to visulize the log file.   


### The command line for docker container/image:  
Path: comp3888_th16_03_christin/mmsegmentation
```
docker build -t m3_image docker/
```
```
docker run --gpus all --shm-size=8g -it -v /home/usyd-05a/data:/mmsegmentation/mmsegmentation/data m3_image
```
### The command line for training and evaluation: 
#### For multiple gpus training format:
```
./tools/dist_train.sh configs/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py {number of gpu >= 2}
```
For instance, to train on 2 gpus:
```
./tools/dist_train.sh configs/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py 2
```
#### For single gpu training:
```
python tools/train.py configs/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py
```
### Exit Docker and Commit image
```
exit
docker commit CONTAINER m3_image
```
### Export the log file form Docker
```
docker cp CONTAINER:/work_dirs/tf_logs/ destination
```
### Tensorboards for visualize scalars
Path: /destination
```
tensorboard --logdir tf_logs --bind_all
```
This will show a link that you can browse at sr9 and port 6006/6007:
http://192.168.10.9:6006 or http://192.168.10.9:6007
# Improvement
Baseline training:  
```
python tools/train.py configs/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py
```
AMP training:  
```
python tools/train.py configs/segformer/segformer_mit-b0_fp16_AMP_8x1_1024x1024_160k_cityscapes.py
```
# Modification

The file ```configs/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py``` specifies the hyperparameters.  
Suggested modifications include ```lr``` (learning rate), ```weight_decay```, ```type``` (optimiser type), ```policy``` (optimiser policy), ```power``` (optimiser power) and ```min_lr``` (minimum lr)  
This is done in the **optimizer** or **lr_config** dict structures.

The file ```configs/_base_/schedules/schedules_160k.py``` specifies the interval (iterations) ONLY. The other present variables, which may have the same name as the above, show no impact on the model  
Interval modifications can be done on the ```runner```, ```checkpoint_config``` and ```evaluation``` dict variables.

file:
```
configs/_base_/default_runtime.py 
``` 
Hooks can enable TextLoggerHook, TensorboardLoggerHook:  
TextLoggerHook for printing the text on console.  
TensorboardLoggerHook for visualize the loss, accuracy and time by using Tensorboard.