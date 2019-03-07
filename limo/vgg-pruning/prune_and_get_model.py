import numpy as np 
import os
import sys
import argparse
import pathlib

import torch
import torch.nn as nn

sys.path.append('/home/ubuntu/zihao/rona_experiments/pytorch-classification') # TODO change it to relative directory
from prune import prune_vgg16_conv_layer, prune_last_fc_layer
import models.cifar as models

parser = argparse.ArgumentParser(description='VGG with mask layer on cifar10')
parser.add_argument('-c', '--prune-candidates', default="./prune_candidate_logs", type=str, help='Directory which stores the prune candidates for each model')
parser.add_argument('-a', '--arch', default='vgg19_bn', type=str, help='The architecture of the trained model')
parser.add_argument('-r', '--resume', default='', type=str, help='The path to the checkpoints')
parser.add_argument('-s', '--save', default='./pruned_models', type=str, help='The path to store the pruned models')
args = parser.parse_args()

num_classes = 10

def update_list(l):
    for i in range(len(l)):
        l[i] -= 1

def main():
    # create pruned model save path
    model_save_directory = os.path.join(args.save, args.arch)
    pathlib.Path(model_save_directory).mkdir(parents=True, exist_ok=True)
    
    # for each class
    for i in range(10):
        print('Pruning class {}'.format(i))
        
        # load pruning candidates
        candidates = np.load(open(os.path.join(args.prune_candidates, 'class_{}.npy'.format(i)), 'rb')).tolist()

        # load cifar10 binary model
        model = models.__dict__[args.arch](num_classes=num_classes)
        model = torch.nn.DataParallel(model).cuda()

        # load checkpoints
        checkpoint = torch.load(os.path.join(args.resume, args.arch, 'checkpoint.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])

        model = model.module

        conv_indices = [idx for idx, (n, p) in enumerate(model.features._modules.items()) if isinstance(p, nn.modules.conv.Conv2d)]

        for layer_index, filter_list in zip(conv_indices, candidates):
            filters_to_remove = list(filter_list)
            sorted(filters_to_remove)
            
            while len(filters_to_remove):
                filter_index = filters_to_remove.pop(0)
                model = prune_vgg16_conv_layer(model, layer_index, filter_index, use_batch_norm=True)
                update_list(filters_to_remove)
        # save the pruned model
        model = prune_last_fc_layer(model, i)
        pruned_model_name = args.arch + '_{}'.format(i) + '_pruned_model.pth'
        torch.save(model, os.path.join(model_save_directory, pruned_model_name))

    print('Pruned model saved at', model_save_directory)

if __name__ == '__main__':
    main()
    