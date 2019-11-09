import re
import glob
import models.cifar as models
from prune import prune_vgg16_conv_layer, prune_last_fc_layer
import numpy as np
import os
import sys
import argparse
import pathlib
import pickle

import torch
import torch.nn as nn

# TODO change it to relative directory
sys.path.append('/home/ubuntu/zihao/rona_experiments/pytorch-classification')

parser = argparse.ArgumentParser(description='VGG with mask layer on cifar10')
parser.add_argument('-c', '--prune-candidates', default="./prune_candidate_logs",
                    type=str, help='Directory which stores the prune candidates for each model')
parser.add_argument('-a', '--arch', default='vgg19_bn',
                    type=str, help='The architecture of the trained model')
parser.add_argument('-r', '--resume', default='', type=str,
                    help='The path to the checkpoints')
parser.add_argument('-s', '--save', default='./pruned_models',
                    type=str, help='The path to store the pruned models')
parser.add_argument('-bce', '--bce', default=False, type=bool,
                    help='Prune according to binary cross entropy loss, i.e. no additional negative output for classifer')
args = parser.parse_args()


path = args.prune_candidates
file_names = [f for f in glob.glob(path + "*.npy", recursive=False)]
group_id_list = [re.search('\(([^)]+)', f_name).group(1)
                 for f_name in file_names]

num_classes = 10  # len(group_id_list)


def main():
    # create pruned model save path
    model_save_directory = os.path.join(args.save, args.arch)
    pathlib.Path(model_save_directory).mkdir(parents=True, exist_ok=True)

    # for each class
    for group_id, file_name in zip(group_id_list, file_names):
        print('Pruning classes {}'.format(group_id))

        # load pruning candidates
        with open(file_name, 'rb') as f:
            candidates = pickle.load(f)

        # load checkpoints
        checkpoint = torch.load(args.resume)
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k.replace('module.', '')] = v
        # model = models.__dict__[args.arch](dataset='cifar10', depth=16, cfg=checkpoint['cfg'])
        model = models.__dict__[args.arch](num_classes=num_classes)
        model.cuda()
        model.load_state_dict(state_dict)
        conv_indices = [idx for idx, (n, p) in enumerate(
            model.features._modules.items()) if isinstance(p, nn.Conv2d)]
        for layer_index, filter_list in zip(conv_indices, candidates):
            filters_to_remove = list(filter_list)
            # filters_to_remove.sort()

            while len(filters_to_remove):
                filter_index = filters_to_remove.pop(0)
                model = prune_vgg16_conv_layer(
                    model, layer_index, filter_index, use_batch_norm=True)
                # update list
                for i in range(len(filters_to_remove)):
                    filters_to_remove[i] -= 1

        # save the pruned model
        group_indices = [int(id) for id in group_id.split("_")]
        model = prune_last_fc_layer(model, group_indices, use_bce=args.bce)
        pruned_model_name = f"{args.arch}_({group_id})_pruned_model.pth"
        torch.save(model, os.path.join(
            model_save_directory, pruned_model_name))

    print('Pruned model saved at', model_save_directory)


if __name__ == '__main__':
    main()
