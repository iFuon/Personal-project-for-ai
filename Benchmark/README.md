# Benchmarking For DDRNet and SegFormer

Automated benchmarking was performed by running Docker containers for each model, and passing configuration files into it to vary combinations of hyperparameters and optimisation settings.
The file ```parameters.txt``` for each model is a .csv file that specifies all such combinations to be tested in the session. Note that permitted variables not specified in these files will be given a default value in the entrypoint test scripts, elaborated below.

To generalise configuration for the above process, .sh file templates are used.

**DDRNet**  
- ddrnet23_slim.yaml (Config for all parameters)

**SegFormer**  
- schedule_160k.py (Config specifying iterations ONLY)  
- segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py (Specify parameters)

### Running

The entrypoint for running a benchmarking session is  
- ```python3 test_ddrnet.py [dataset] [repetition]``` [DDRNet]  
- ```python3 test.py [dataset] [repetition]``` [SegFormer]

Use ```python3 [file] /home/usyd-05a/data 3``` for default situations.

The ```[dataset]``` specifies the path of the dataset. This is **/home/usyd-05a/data** for the personal server, but should be changed to suit deployment (with no / at the end).  
```[repetition]``` specifies a positive integer for the number of repetitions for each combination of parameters.txt

## Optimisation Settings

SegFormer only has AMP enabled, while DDRNet has both AMP and Channel.Last  
These settings are to be manually altered in the ```# Execute the training subroutine in Docker``` line of the entrypoint Python scripts

**DDRNet**  
- Modify the ```--benchmark n``` clause  
- n = 1 (AMP), n = 2 (Base), n = 3 (Channel.Last), n = 4 (AMP + Channel.Last)

**SegFormer**  
- Modify the configuration path clause  
- ```configs/segformer/segformer_mit-b0_fp16_AMP_8x1_1024x1024_160k_cityscapes.py```  [AMP]  
- ```configs/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py``` [Base]

Also modify the ``in_hyperparameter_config`` and ``out_hyperparameter_config`` variables at the start of the file to be the same as the above path clause.

### Results

Once all tests are complete without interruption, a log file will be exported into the Benchmarking directory, in the same sequence of the combinations, including repetitions of each.  
- ```log``` [DDRNet]  
- ```work_dirs``` [SegFormer]

The ```tf_logs``` folder can subsequently be used with TensorBoard

-----------------------------------------------------------------------------------------------------

# Benchmarking For Other Models

https://bitbucket.org/kiriyachristin/comp3888_th16_03_repom1/src/master/ [DecoupleSegNets]  
https://bitbucket.org/abstractblaze/hanet/src/master/ [HANet]
