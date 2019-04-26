import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
from cifar10class import *
import random
import math 

def loader(class_index = None, batch_size=64, num_workers=2, pin_memory=True, group = None, activations = False, multiplier = 1):
    """
    Modified version of the training set loader. If activations is specified, generates samples from
    each class within the activations list
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if activations and group:
        samples = []
        for class_idx in group:
             samples.extend(random.sample(get_class_i(x_train, y_train, class_idx), math.ceil(1000 / len(group))))
        samples = [samples]
    elif group:
        samples = [get_random_images(x_train, y_train, *group)]
        for class_idx in group:
            samples.append(get_class_i(x_train, y_train, class_idx) * multiplier)
        for x in samples:
            print(len(x))
        input()
    else:
        samples = [get_random_images(x_train, y_train, class_index), get_class_i(x_train, y_train, class_index)]
 
    training_data = DatasetMaker(samples, transform_with_aug)
    
    return data.DataLoader(training_data,
        batch_size=batch_size,
        shuffle = True if activations == False else False,
        num_workers=num_workers,
        pin_memory=pin_memory)

def test_loader(class_index = None, batch_size=64, num_workers=2, pin_memory=True, group = None):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if group:
        samples = [get_random_images(x_test, y_test, *group)]
        for class_idx in group:
            samples.append(get_class_i(x_test, y_test, class_idx))
    else:
        samples = [get_random_images(x_test, y_test, class_index), get_class_i(x_test, y_test, class_index)]
    test_data = DatasetMaker(
            samples, transform_no_aug)
    return data.DataLoader(test_data,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=num_workers,
                           pin_memory=pin_memory)
