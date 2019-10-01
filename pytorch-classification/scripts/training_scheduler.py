import torch
import threading
from subprocess import run
import multiprocessing as mp

'''
1) initialize bounded producer/consumer queue of size max(num_devices (param), output from torch.cuda.device_count())
'''
def train(executables, allowable_devices=range(torch.cuda.device_count())):
    free_devices = set(allowable_devices)
    cv = threading.Condition()
    for executable in executables:
        cv.acquire()
        while len(free_devices) == 0:
            cv.wait()
        assigned_device = free_devices.pop()
        print(assigned_device)
        mp.Process(target=execute_on_device, args=(assigned_device, executable, cv, free_devices)).start()
        cv.release()

def execute_on_device(GPU_ID, executable, cond_var, free_devices):
    # train the model
    run(['CUDA_VISIBLE_DEVICES='+GPU_ID] + executable.split(" "))
    # mark this GPU as free
    free_devices.push(GPU_ID)
    # signal that this model has finished training (wakes up a waiting model)
    cond_var.signal()



if __name__ == '__main__':
    to_train = [
        "python3 cifar_binary.py --pruned -a vgg19_bn --epochs 41 --schedule 20 30 --gamma 0.1 --resume ./apoz_pruned_models/vgg19_bn/vgg19_bn_0_pruned_model.pth --checkpoint ./apoz_pruned_retrain/vgg19_bn/vgg19_bn_0_pruned_model --train-batch 64 --class-index 0",
        "python3 cifar_binary.py --pruned -a vgg19_bn --epochs 41 --schedule 20 30 --gamma 0.1 --resume ./apoz_pruned_models/vgg19_bn/vgg19_bn_1_pruned_model.pth --checkpoint ./apoz_pruned_retrain/vgg19_bn/vgg19_bn_1_pruned_model --train-batch 64 --class-index 1",
        "python3 cifar_binary.py --pruned -a vgg19_bn --epochs 41 --schedule 20 30 --gamma 0.1 --resume ./apoz_pruned_models/vgg19_bn/vgg19_bn_2_pruned_model.pth --checkpoint ./apoz_pruned_retrain/vgg19_bn/vgg19_bn_2_pruned_model --train-batch 64 --class-index 2",
        "python3 cifar_binary.py --pruned -a vgg19_bn --epochs 41 --schedule 20 30 --gamma 0.1 --resume ./apoz_pruned_models/vgg19_bn/vgg19_bn_3_pruned_model.pth --checkpoint ./apoz_pruned_retrain/vgg19_bn/vgg19_bn_3_pruned_model --train-batch 64 --class-index 3",
        "python3 cifar_binary.py --pruned -a vgg19_bn --epochs 41 --schedule 20 30 --gamma 0.1 --resume ./apoz_pruned_models/vgg19_bn/vgg19_bn_4_pruned_model.pth --checkpoint ./apoz_pruned_retrain/vgg19_bn/vgg19_bn_4_pruned_model --train-batch 64 --class-index 4",
        "python3 cifar_binary.py --pruned -a vgg19_bn --epochs 41 --schedule 20 30 --gamma 0.1 --resume ./apoz_pruned_models/vgg19_bn/vgg19_bn_5_pruned_model.pth --checkpoint ./apoz_pruned_retrain/vgg19_bn/vgg19_bn_5_pruned_model --train-batch 64 --class-index 5",
        "python3 cifar_binary.py --pruned -a vgg19_bn --epochs 41 --schedule 20 30 --gamma 0.1 --resume ./apoz_pruned_models/vgg19_bn/vgg19_bn_6_pruned_model.pth --checkpoint ./apoz_pruned_retrain/vgg19_bn/vgg19_bn_6_pruned_model --train-batch 64 --class-index 6",
        "python3 cifar_binary.py --pruned -a vgg19_bn --epochs 41 --schedule 20 30 --gamma 0.1 --resume ./apoz_pruned_models/vgg19_bn/vgg19_bn_7_pruned_model.pth --checkpoint ./apoz_pruned_retrain/vgg19_bn/vgg19_bn_7_pruned_model --train-batch 64 --class-index 7",
        "python3 cifar_binary.py --pruned -a vgg19_bn --epochs 41 --schedule 20 30 --gamma 0.1 --resume ./apoz_pruned_models/vgg19_bn/vgg19_bn_8_pruned_model.pth --checkpoint ./apoz_pruned_retrain/vgg19_bn/vgg19_bn_8_pruned_model --train-batch 64 --class-index 8",
        "python3 cifar_binary.py --pruned -a vgg19_bn --epochs 41 --schedule 20 30 --gamma 0.1 --resume ./apoz_pruned_models/vgg19_bn/vgg19_bn_9_pruned_model.pth --checkpoint ./apoz_pruned_retrain/vgg19_bn/vgg19_bn_9_pruned_model --train-batch 64 --class-index 9",
    ]
    train(to_train)