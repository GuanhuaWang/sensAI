import numpy as np 
import os
import sys
import argparse
import pathlib
import copy

import torch
import torch.nn as nn

sys.path.append('/home/ubuntu/zihao/rona_experiments/pytorch-classification') # TODO change it to relative directory
from res_prune import *
import models as models

parser = argparse.ArgumentParser(description='VGG with mask layer on cifar10')
parser.add_argument('-c', '--prune-candidates', default="./activations/prune_candidate_logs", type=str, help='Directory which stores the prune candidates for each model')
parser.add_argument('-a', '--arch', default='vgg19_bn', type=str, help='The architecture of the trained model')
parser.add_argument('-r', '--resume', default='', type=str, help='The path to the checkpoints')
parser.add_argument('-s', '--save', default='./pruned_models', type=str, help='The path to store the pruned models')
parser.add_argument('-d', '--depth', default=164, type=int, help='Depth of neural net')
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
        candidates = np.load(open(os.path.join(args.prune_candidates, 'class_{}_apoz_layer_thresholds.npy'.format(i)), 'rb')).tolist()
#        print(candidates)

        # load cifar10 binary model
#        model = models.__dict__[args.arch](num_classes=num_classes)
#        model = torch.nn.DataParallel(model).cuda()

        # load checkpoints
        checkpoint = torch.load(os.path.join(args.resume, 'checkpoint.pth.tar'))
        model = models.__dict__[args.arch](depth=args.depth)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.cuda()
        
        new_model = copy.deepcopy(model)
#        model = torch.nn.DataParallel(model).cuda()

#        model = model.modules
#        print(list(model.children()))
#        print(len(list(model.children())))
#        print(list(model.modules()))

        conv_indices = [idx for idx, m in enumerate(new_model.modules()) if isinstance(m,  nn.Conv2d)]

        # don't prune first conv layer and downsample layers
        downsamples = [1, 13, 178, 343]
        for d in downsamples:
            conv_indices.remove(d)
        # don't prune last relu activation (no corresponding conv layer)
        candidates.pop(len(candidates)-1)

        # only prune out channels of first 2 conv layers per block
        last_conv = 2
        while last_conv < len(conv_indices):
            conv_indices.pop(last_conv)
            last_conv += 2

        # separate first relu layer and the last 2 in each block
        first_relu_candidates = []
        first_relu = 0
        while first_relu < len(candidates):
            first_relu_candidates.append(candidates.pop(first_relu))
            first_relu += 2
        print(len(conv_indices))
        print(len(first_relu_candidates))
#        print(conv_indices)
        print(len(candidates))
#        print(list(model.modules())[3])
        for layer_index, filter_list in zip(conv_indices, candidates):
            # prune input channels for first conv layer
            conv_no = conv_indices.index(layer_index)
            if conv_no % 2 == 0:
                filters_to_remove = first_relu_candidates[int(conv_no/2)]
                sorted(filters_to_remove)
                new_model = prune_selection_layer(new_model, layer_index, filters_to_remove)

                while len(filters_to_remove):
                    filter_index = filters_to_remove.pop(0)
                    new_model = prune_first_conv_layer(new_model, layer_index, filter_index)
                    update_list(filters_to_remove)

            filters_to_remove = list(filter_list)
            sorted(filters_to_remove)
            
            while len(filters_to_remove):
                filter_index = filters_to_remove.pop(0)
                new_model = prune_resnet_conv_layer(new_model, layer_index, filter_index, use_batch_norm=True)
                update_list(filters_to_remove)
        # save the pruned model
#        print(model.fc.bias.data)
        new_model = prune_last_fc_layer(new_model, i)
#        model = prune_downsampling(model)
#        print(model.fc.bias.data)
        pruned_model_name = args.arch + '_{}'.format(i) + '_pruned_model.pth'
        torch.save(new_model, os.path.join(model_save_directory, pruned_model_name))

    print('Pruned model saved at', model_save_directory)

if __name__ == '__main__':
    main()
    
