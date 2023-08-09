#!/usr/bin/python

import os
import sys
import subprocess
import random
import time

in_iteration_config = "schedule_160k.sh"
out_iteration_config = "schedule_160k.py"

# Modify the hyperparameter configuration files in accordance to the optimisation setting config substitution (in the run section)
in_hyperparameter_config = "segformer_mit-b0_8x1_1024x1024_160k_cityscapes.sh"
out_hyperparameter_config = "segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py"
# "segformer_mit-b0_fp16_AMP_8x1_1024x1024_160k_cityscapes" [AMP]
# "segformer_mit-b0_8x1_1024x1024_160k_cityscapes" [Base]

os.environ['TYPE'] = 'AdamW' # 'AdamW'
os.environ['LR'] = '0.00006' # 0.00006
os.environ['WEIGHT_DECAY'] = '0.01' # 0.01

os.environ['POLICY'] = 'poly' # 'poly'
os.environ['POWER'] = '1.0' # 1.0
os.environ['MIN_LR'] = '0' # 0

os.environ['INTERVAL'] = '600' # 16000 (Minimum number is 600)

# List of all parameters to be varied
environment_variables = ['TYPE', 'LR', 'WEIGHT_DECAY', 'POLICY', 'POWER', 'MIN_LR'] # Omit Interval due to Dual Configs

# Directory of the benchmark files
work_directory = os.path.realpath(os.path.dirname(__file__))

# -----------------------------------------------------------------------------------------------------

# Creates a random string to serve as a unique image name
def random_string():
    pool = list("abcdefghijklmnopqrstuvwxyz")
    string = ""
    for i in range(0, 5):
        string += random.choice(pool)
    return string

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 test.py [dataset] [repetitions]")
        exit(0)
    try:   
        repetitions = int(sys.argv[2])       
        if repetitions <= 0:
              print("[repetitions] must be positive")
              exit(0)
    except Exception:
        print("[repetitions] must an integer")
        exit(0)
    
    # name = random_string()
    name = "testimage"
    
    r = subprocess.run("docker run --name {} --gpus all --shm-size=8g -it -d -v {}:/mmsegmentation/mmsegmentation/data m3_image"
                       .format(name, sys.argv[1]).split()) # Detached
    if r.returncode != 0:
        exit(0)
    
    # Obtain Image Identifier
    i = subprocess.Popen("docker ps".split(), stdout = subprocess.PIPE)
    i.wait()
    j = subprocess.Popen("grep {}".format(name).split(), stdin = i.stdout, stdout = subprocess.PIPE)
    j.wait()
    k = subprocess.Popen(["awk", "{print $1}"], stdin = j.stdout, stdout = subprocess.PIPE)
    k.wait()
    identifier = k.stdout.read().decode("utf-8").rstrip()
    print("Image Identifier: {}".format(identifier))

    # -----------------------------------------------------------------------------------------------------
     
    # Read parameters.txt for varying model elements
    with open("parameters.txt", "r") as f:
        document = f.readlines()
        
        var = document[0].upper().rstrip().split(",")
        selected_variables = list(var)

    # Train for every line of the parameter config file
    i = 1
    while i < len(document):
        line = document[i].rstrip().split(",")
        j = 0
        while j < len(line): # Alter environment variables
            os.environ[selected_variables[j]] = line[j] 
            j += 1
        i += 1
        
        for repeat in range(repetitions): # Repeat
            # -----------------------------------------------------------------------------------------------------

            # Access the interval value
            process = subprocess.Popen(["bash", in_iteration_config] + [os.environ["INTERVAL"]], 
                                                stdout = subprocess.PIPE)
            process.wait()
            with open(out_iteration_config, "w") as f: # Write the interval config
                f.write(process.stdout.read().decode("utf-8"))
            
            
            
            # Use the in_hyperparameter template with environment variables, and write to the out_config
            process = subprocess.Popen(["bash", in_hyperparameter_config] + [os.environ[i] for i in environment_variables], 
                                    stdout = subprocess.PIPE)    
            process.wait() # bash segformer_mit-b0_8x1_1024x1024_160k_cityscapes.sh 'AdamW' '0.00006' '0.01' 'poly' '1.0' '0'

            with open(out_hyperparameter_config, "w") as f:
                f.write(process.stdout.read().decode("utf-8"))

            # Copy interval config to the model's configuration location in Docker
            subprocess.run("docker cp {}/{} {}:/mmsegmentation/mmsegmentation/configs/_base_/schedules/"
                              .format(work_directory, out_iteration_config, identifier).split())

            # Copy hyperparam config in the same manner
            subprocess.run("docker cp {}/{} {}:/mmsegmentation/mmsegmentation/configs/segformer/"
                              .format(work_directory, out_hyperparameter_config, identifier).split())
            
            # Execute the training subroutine in Docker
            subprocess.run("docker exec {} python tools/train.py configs/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py"
                               .format(identifier).split()) # , stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
            
            # "docker exec {} python tools/train.py configs/segformer/segformer_mit-b0_fp16_AMP_8x1_1024x1024_160k_cityscapes.py" [AMP]
            # "docker exec {} python tools/train.py configs/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py" [BASE]
              
    # -----------------------------------------------------------------------------------------------------
    
    # Delete the current work_dirs in the source
    subprocess.run("rm -r {}/work_dirs/".format(work_directory).split(), stdout=subprocess.DEVNULL,
                       stderr=subprocess.STDOUT)
    
    # Copy the model result output from Docker to the benchmarking directory
    subprocess.run("docker cp {}:/mmsegmentation/mmsegmentation/work_dirs/ {}/"
                       .format(identifier, work_directory).split())
    
    # End the docker container
    subprocess.run("docker stop {}".format(name).split(), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    subprocess.run("docker rm {}".format(name).split(), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
