import numpy as np 
import os
import sys
import argparse
import pathlib

import torch
import torch.nn as nn
import torchvision.models as models

sys.path.append('/home/ubuntu/zihao/rona_experiments/pytorch-classification') # TODO change it to relative directory
from prune import prune_vgg16_conv_layer, prune_last_fc_layers
# import models.cifar as models

parser = argparse.ArgumentParser(description='VGG with mask layer on cifar10')
parser.add_argument('-c', '--prune-candidates', default="./prune_candidate_logs", type=str, help='Directory which stores the prune candidates for each model')
parser.add_argument('-a', '--arch', default='vgg19_bn', type=str, help='The architecture of the trained model')
parser.add_argument('-r', '--resume', default='', type=str, help='The path to the checkpoints')
parser.add_argument('-s', '--save', default='./pruned_models', type=str, help='The path to store the pruned models')
parser.add_argument('-bce', '--bce', default=False, type=bool, help='Prune according to binary cross entropy loss, i.e. no additional negative output for classifer')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
args = parser.parse_args()



import glob
import re

path = args.prune_candidates
file_names = [f for f in glob.glob(path + "group_*.npy", recursive=False)]
group_id_list = [re.search('\(([^)]+)', f_name).group(1) for f_name in file_names] 

num_classes = 10 # len(group_id_list)

def update_list(l):
    for i in range(len(l)):
        l[i] -= 1

def main():
    # create pruned model save path
    model_save_directory = os.path.join(args.save, args.arch)
    pathlib.Path(model_save_directory).mkdir(parents=True, exist_ok=True)

    #np.save(open("prune_candidate_logs/grouping_config.npy", "wb"), groups)
    groups = np.load(open(path + "grouping_config.npy", "rb"))
    np.save(open(os.path.join(args.save, "grouping_config.npy"), "wb"), groups)
   
    # for each class
    for group_id, file_name in zip(group_id_list, file_names):
        print('Pruning classes {} from candidates in {}'.format(group_id, file_name)) 
        group_indices = groups[int(group_id)]
        # load pruning candidates
        candidates =  np.load(open(file_name, 'rb')).tolist()

        # load checkpoints
        # checkpoint = torch.load(args.resume)
        # model = models.__dict__[args.arch](dataset='cifar10', depth=16, cfg=checkpoint['cfg'])
        if args.pretrained:
            # print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
            model = torch.nn.DataParallel(model).cuda()
        else:
            checkpoint = torch.load(args.resume)
            model = models.__dict__[args.arch](num_classes=num_classes)
            model = torch.nn.DataParallel(model).cuda()
            model.load_state_dict(checkpoint['state_dict'])

        model = model.module

        conv_indices = [idx for idx, (n, p) in enumerate(model.features._modules.items()) if isinstance(p, nn.modules.conv.Conv2d)]
        
        # print(list(enumerate(zip(conv_indices, candidates)))) 
        offset = 0
        for layer_index, filter_list in zip(conv_indices, candidates):
            offset += 1
            filters_to_remove = list(filter_list)
            sorted(filters_to_remove)
                       
            while len(filters_to_remove):
                filter_index = filters_to_remove.pop(0)
                model = prune_vgg16_conv_layer(model, layer_index, filter_index, use_batch_norm=True)
                update_list(filters_to_remove)
            
        # save the pruned model
        group_indices = groups[int(group_id)]
        # The input dimension of the first fc layer is pruned from above
        model = prune_last_fc_layers(model, \
                                     group_indices, \
                                     filter_indices = candidates[offset:], \
                                     use_bce = args.bce)
  
        pruned_model_name = args.arch + '_({})'.format(group_id) + '_pruned_model.pth'
        print('Grouped mode  %s Total params: %.2fM' % (group_id ,sum(p.numel() for p in model.parameters())/1000000.0))
        torch.save(model, os.path.join(model_save_directory, pruned_model_name))
    print('Pruned model saved at', model_save_directory)

if __name__ == '__main__':
    main()
    
