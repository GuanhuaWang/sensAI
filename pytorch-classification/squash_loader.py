import numpy as np
import os
import sys
import argparse
import pathlib

import torch
import torch.nn as nn
import models.cifar as models

sys.path.append('/home/ubuntu/guanhua/sensai_experiments/pytorch-classification')

parser = argparse.ArgumentParser(description='VGG with mask layer on cifar10')
parser.add_argument('-c', '--prune-candidates', default="./squash_logs", type=str, help='Get squashed value for each feature-map')
parser.add_argument('-a', '--arch', default='vgg19_bn', type=str, help='The architecture of the trained model')
parser.add_argument('-r', '--resume', default='', type=str, help='The path to the checkpoints')
parser.add_argument('-s', '--save', default='./pruned_models', type=str, help='The path to store the pruned models')
args = parser.parse_args()

num_classes = 10

if __name__ == '__main__':
    print('start ')
    for i in range(10):
        print('Pruning class {}'.format(i))
        candidates = np.load(open(os.path.join(args.prune_candidates, 'class_{}_lc.npy'.format(i)), 'rb')).tolist()

        for filter_list in candidates:
            print('{}'.format(filter_list.shape))
