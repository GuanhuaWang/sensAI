import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
from cifar10class import *
import random
import math


def loader(class_index=None, batch_size=64, num_workers=2, pin_memory=True, group=None, activations=False, multiplier=1):
    """
    Modified version of the training set loader. If activations is specified, generates samples from
    each class within the activations list
    """
    if activations and group:
        samples = []
        for class_idx in group:
            samples.extend(random.sample(
                get_class_i(x_train, y_train, class_idx), 500))
        samples = [samples]
    elif group:
        samples = [get_random_images(x_train, y_train, *group)]
        for class_idx in group:
            samples.append(get_class_i(
                x_train, y_train, class_idx) * multiplier)
    else:
        samples = [get_random_images(x_train, y_train, class_index), get_class_i(
            x_train, y_train, class_index)]

    training_data = DatasetMaker(samples, transform_with_aug)

    return data.DataLoader(training_data,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory)


def test_loader(class_index=None, batch_size=64, num_workers=2, pin_memory=True, group=None):
    if group:
        samples = [get_random_images(x_test, y_test, *group)]
        for class_idx in group:
            samples.append(get_class_i(x_test, y_test, class_idx))
    else:
        samples = [get_random_images(x_test, y_test, class_index), get_class_i(
            x_test, y_test, class_index)]
    test_data = DatasetMaker(samples, transform_no_aug)
    return data.DataLoader(test_data,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=num_workers,
                           pin_memory=pin_memory)
