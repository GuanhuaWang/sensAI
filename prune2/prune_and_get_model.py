import re
import glob
import models.cifar as models
from prune_utils.prune_vgg16 import prune_vgg16_conv_layer, prune_last_fc_layer
import numpy as np
import os
import sys
import argparse
import pathlib
import pickle

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import load_model
from prune_utils import prune_resnet, prune_vgg16, layer_prune

import copy


parser = argparse.ArgumentParser(description='VGG with mask layer on cifar10')
parser.add_argument('-d', '--dataset', required=True, type=str)
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


def update_list(l):
    for i in range(len(l)):
        l[i] -= 1


def prune_vgg_main(model, candidates, group_indices):
    conv_indices = [idx for idx, (n, p) in enumerate(
        model.features._modules.items()) if isinstance(p, nn.Conv2d)]
    for layer_index, filter_list in zip(conv_indices, candidates):
        filters_to_remove = list(filter_list)
        # filters_to_remove.sort()

        while len(filters_to_remove):
            filter_index = filters_to_remove.pop(0)
            model = prune_vgg16.prune_vgg16_conv_layer(
                model, layer_index, filter_index, use_batch_norm=True)
            # update list
            for i in range(len(filters_to_remove)):
                filters_to_remove[i] -= 1

    model = prune_vgg16.prune_last_fc_layer(
        model, group_indices, use_bce=args.bce)
    return model


def prune_resnet_main(model, candidates, group_indices):
    conv_indices = [idx for idx, m in enumerate(
        model.modules()) if isinstance(m, nn.Conv2d)]

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
    # print(conv_indices)
    print(len(candidates))
    # print(list(model.modules())[3])
    for layer_index, filter_list in zip(conv_indices, candidates):
        # prune input channels for first conv layer
        conv_no = conv_indices.index(layer_index)
        if conv_no % 2 == 0:
            filters_to_remove = first_relu_candidates[int(conv_no/2)]
            sorted(filters_to_remove.sort())
            model = prune_resnet.prune_selection_layer(
                model, layer_index, filters_to_remove)

            while len(filters_to_remove):
                filter_index = filters_to_remove.pop(0)
                model = prune_resnet.prune_first_conv_layer(
                    model, layer_index, filter_index)
                update_list(filters_to_remove)

        filters_to_remove = list(filter_list)
        sorted(filters_to_remove)

        while len(filters_to_remove):
            filter_index = filters_to_remove.pop(0)
            model = prune_resnet.prune_resnet_conv_layer(
                model, layer_index, filter_index, use_batch_norm=True)
            update_list(filters_to_remove)

    if model.fc is None:
        raise ValueError("No linear layer found in classifier")
    layer_prune.prune_linear_layer(model.fc, group_indices)
    return model


def main():
    file_names = [f for f in glob.glob(
        args.prune_candidates + "*.npy", recursive=False)]
    group_id_list = [re.search('\(([^)]+)', f_name).group(1)
                     for f_name in file_names]

    use_cuda = torch.cuda.is_available()

    print(f'==> Preparing dataset {args.dataset}')
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    else:
        raise NotImplementedError

    # create pruned model save path
    model_save_directory = os.path.join(args.save, args.arch)
    pathlib.Path(model_save_directory).mkdir(parents=True, exist_ok=True)

    # for each class
    for group_id, file_name in zip(group_id_list, file_names):
        # load pruning candidates
        with open(file_name, 'rb') as f:
            candidates = pickle.load(f)

        # load checkpoints
        model = load_model.load_pretrain_model(
            args.arch, args.dataset, args.resume, num_classes, use_cuda)
        new_model = copy.deepcopy(model)

        group_indices = [int(id) for id in group_id.split("_")]
        # pruning
        if args.arch.startswith('vgg'):
            model = prune_vgg_main(new_model, candidates, group_indices)
        elif args.arch.startswith('resnet'):
            model = prune_resnet_main(new_model, candidates, group_indices)
        else:
            raise NotImplementedError

        # save the pruned model
        pruned_model_name = f"{args.arch}_({group_id})_pruned_model.pth"
        torch.save(model, os.path.join(
            model_save_directory, pruned_model_name))
        print('Pruned model saved at', model_save_directory)


if __name__ == '__main__':
    main()
