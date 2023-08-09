#!/usr/bin/python

import os
import sys
import subprocess
import random
import time

in_config = "ddrnet23_slim.sh"
out_config = "ddrnet23_slim.yaml"

os.environ['BATCH_SIZE_PER_GPU'] = '8' # '8'
os.environ['LR'] = '0.01' # Learning Rate, 0.01
os.environ['WD'] = '0.0005' # Weight Decay, 0.0005
os.environ['MOMENTUM'] = '0.9' # 0.9

os.environ['END_EPOCH'] = '1' # Interavals, 484

# List of all parameters to be varied, in order
environment_variables = ['BATCH_SIZE_PER_GPU', 'END_EPOCH', 'LR', 'WD', 'MOMENTUM']

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
        print("Usage: python3 test_ddrnet.py [dataset] [repetitions]")
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
    name = "testimageddrnet"
        
    r = subprocess.run("docker run -w /m4_DDRNet/m4_DDRNet --name {} --gpus all --shm-size=8g -it -d -v {}:/m4_DDRNet/m4_DDRNet/data ddr_m4"
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

            # Use in_config template with environment variables, and write to the out_config  
            process = subprocess.Popen(["bash", in_config] + [os.environ[i] for i in environment_variables], 
                                       stdout = subprocess.PIPE)   
            process.wait() # bash ddrnet23_slim.yaml.sh '8' '1 'sgd' '0.01' '0.0005' '0.9'

            with open(out_config, "w") as f:
                f.write(process.stdout.read().decode("utf-8"))

            # Copy out_config to the model's configuration location in Docker
            subprocess.run("docker cp {}/{} {}:/m4_DDRNet/m4_DDRNet/experiments/cityscapes/"
                              .format(work_directory, out_config, identifier).split())

            # Execute the training subroutine in Docker
            subprocess.run("docker exec {} python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --cfg experiments/cityscapes/{} --benchmark 2"
                               .format(identifier, out_config).split()) # , stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
            
            # --benchmark 1 is AMP
            # --benchmark 2 is Base
            # --benchmark 3 is Channel.Last
            # --benchmark 4 is AMP + Channel.Last
    
    # -----------------------------------------------------------------------------------------------------
    
    # Delete the current log folder in the source
    subprocess.run("rm -r {}/log/".format(work_directory).split(), stdout=subprocess.DEVNULL,
                       stderr=subprocess.STDOUT)
    
    # Copy the model LOG from Docker to the benchmarking directory
    subprocess.run("docker cp {}:/m4_DDRNet/m4_DDRNet/log {}/"
                       .format(identifier, work_directory).split())
       
    # Copy the model OUTPUT from Docker to the benchmarking directory
    subprocess.run("docker cp {}:/m4_DDRNet/m4_DDRNet/output {}/log/"
                       .format(identifier, work_directory).split())
    
    # End the docker container
    subprocess.run("docker stop {}".format(name).split(), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    subprocess.run("docker rm {}".format(name).split(), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
