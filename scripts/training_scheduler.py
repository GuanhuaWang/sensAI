import torch
import threading
import subprocess
import multiprocessing as mp
import os

pruned_model_path="./pruned_models/vgg19_bn/"
retrained_model_path="./retrained_model/vgg19_bn/"
'''
1) initialize bounded producer/consumer queue of size max(num_devices (param), output from torch.cuda.device_count())
'''
def train(executables, allowable_devices=range(torch.cuda.device_count())):
    free_devices = mp.Queue(maxsize=len(allowable_devices))
    for i in allowable_devices:
        free_devices.put(i)
    for executable in executables:
        assigned_device = free_devices.get()
        print("script: '" + str(executable) + "' assigned to GPU: " + str(assigned_device))
        mp.Process(target=execute_on_device, args=(assigned_device, executable, free_devices)).start()

def execute_on_device(GPU_ID, executable, free_devices):
    # train the model
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
    executable_tokens = executable.split(" ")
    stdout_file = None
    if ">" in executable_tokens:
        idx = executable_tokens.index(">")
        stdout_file = open(executable_tokens[idx+1], "w")
        executable_tokens = executable_tokens[:idx]
    print(stdout_file)
    subprocess.run(executable_tokens, stdout=stdout_file)
    # mark this GPU as free
    free_devices.put(GPU_ID)
    if stdout_file is not None:
        stdout_file.close()

def get_stdout(executable_tokens):
    if '>' in executable_tokens:
        return executable
    else:
        return None    

if __name__ == '__main__':
    to_train = [
        "python3 cifar_binary.py --pruned -a vgg19_bn --lr 0.01 --epochs 40 --schedule 20 30 --gamma 0.1 --resume "+pruned_model_path+"vgg19_bn_0_pruned_model.pth --checkpoint "+retrained_model_path+"vgg19_bn_0_pruned_model --train-batch 64 --class-index 0",
        "python3 cifar_binary.py --pruned -a vgg19_bn --lr 0.01 --epochs 40 --schedule 20 30 --gamma 0.1 --resume "+pruned_model_path+"vgg19_bn_1_pruned_model.pth --checkpoint "+retrained_model_path+"vgg19_bn_1_pruned_model --train-batch 64 --class-index 1",
        "python3 cifar_binary.py --pruned -a vgg19_bn --lr 0.01 --epochs 40 --schedule 20 30 --gamma 0.1 --resume "+pruned_model_path+"vgg19_bn_2_pruned_model.pth --checkpoint "+retrained_model_path+"vgg19_bn_2_pruned_model --train-batch 64 --class-index 2",
        "python3 cifar_binary.py --pruned -a vgg19_bn --lr 0.01 --epochs 40 --schedule 20 30 --gamma 0.1 --resume "+pruned_model_path+"vgg19_bn_3_pruned_model.pth --checkpoint "+retrained_model_path+"vgg19_bn_3_pruned_model --train-batch 64 --class-index 3",
        "python3 cifar_binary.py --pruned -a vgg19_bn --lr 0.01 --epochs 40 --schedule 20 30 --gamma 0.1 --resume "+pruned_model_path+"vgg19_bn_4_pruned_model.pth --checkpoint "+retrained_model_path+"vgg19_bn_4_pruned_model --train-batch 64 --class-index 4",
        "python3 cifar_binary.py --pruned -a vgg19_bn --lr 0.01 --epochs 40 --schedule 20 30 --gamma 0.1 --resume "+pruned_model_path+"vgg19_bn_5_pruned_model.pth --checkpoint "+retrained_model_path+"vgg19_bn_5_pruned_model --train-batch 64 --class-index 5",
        "python3 cifar_binary.py --pruned -a vgg19_bn --lr 0.01 --epochs 40 --schedule 20 30 --gamma 0.1 --resume "+pruned_model_path+"vgg19_bn_6_pruned_model.pth --checkpoint "+retrained_model_path+"vgg19_bn_6_pruned_model --train-batch 64 --class-index 6",
        "python3 cifar_binary.py --pruned -a vgg19_bn --lr 0.01 --epochs 40 --schedule 20 30 --gamma 0.1 --resume "+pruned_model_path+"vgg19_bn_7_pruned_model.pth --checkpoint "+retrained_model_path+"vgg19_bn_7_pruned_model --train-batch 64 --class-index 7",
        "python3 cifar_binary.py --pruned -a vgg19_bn --lr 0.01 --epochs 40 --schedule 20 30 --gamma 0.1 --resume "+pruned_model_path+"vgg19_bn_8_pruned_model.pth --checkpoint "+retrained_model_path+"vgg19_bn_8_pruned_model --train-batch 64 --class-index 8",
        "python3 cifar_binary.py --pruned -a vgg19_bn --lr 0.01 --epochs 40 --schedule 20 30 --gamma 0.1 --resume "+pruned_model_path+"vgg19_bn_9_pruned_model.pth --checkpoint "+retrained_model_path+"vgg19_bn_9_pruned_model --train-batch 64 --class-index 9",
    ]
    train(to_train)
