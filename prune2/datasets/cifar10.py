import torch
from torchvision import datasets
from torchvision import transforms
from typing import List, Tuple

from datasets import utils


# Transformations
RC = transforms.RandomCrop(32, padding=4)
RHF = transforms.RandomHorizontalFlip()
RVF = transforms.RandomVerticalFlip()
NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
TT = transforms.ToTensor()
TPIL = transforms.ToPILImage()

# Transforms object for trainset with augmentation
transform_with_aug = transforms.Compose([TPIL, RC, RHF, TT, NRM])
# Transforms object for testset with NO augmentation
transform_no_aug = transforms.Compose([TT, NRM])


CIFAR10_DATASET_ROOT = './data/'


class CIFAR10TrainingSetWrapper(utils.TrainingSetWrapper):
    def __init__(self, class_group: Tuple[int], negative_samples=False):
        dataset = datasets.CIFAR10(root=CIFAR10_DATASET_ROOT, train=True,
                                   download=True, shuffle=False, transforms=transform_with_aug)
        super().__init__(dataset, class_group, negative_samples)
