import re
import glob
import models.cifar as models
import os
import sys
import argparse
import pathlib
import pickle

import torch
from torch import nn
import load_model

from prune_utils.layer_prune import (
    prune_output_linear_layer_,
    prune_contiguous_conv2d_,
    prune_conv2d_out_channels_,
    prune_batchnorm2d_,
    prune_linear_in_features_)
from models.cifar.resnet import Bottleneck

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


def prune_vgg(model, pruned_candidates, group_indices):
    features = model.features
    conv_indices = [i for i, layer in enumerate(features) if isinstance(layer, nn.Conv2d)]
    conv_bn_indices = [i for i, layer in enumerate(features) if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d))]
    assert len(conv_indices) == len(pruned_candidates)
    assert len(conv_indices) * 2 == len(conv_bn_indices)

    for i, conv_index in enumerate(conv_indices):
        next_conv = None
        for j in range(conv_index + 1, len(features)):
            l = conv_indices[j]
            if isinstance(l, nn.Conv2d):
                next_conv = l
                break
        if next_conv is None:
            break
        bn = model.features[conv_index + 1]
        assert isinstance(bn, nn.BatchNorm2d)
        prune_contiguous_conv2d_(
            features[conv_index],
            next_conv,
            pruned_candidates[i],
            bn=bn)

    # Prunning the last conv layer. This affects the first linear layer of the classifier.
    last_conv = features[conv_indices[-1]]
    pruned_indices = pruned_candidates[-1]
    prune_conv2d_out_channels_(last_conv, pruned_indices)
    prune_batchnorm2d_(features[conv_bn_indices[-1]], pruned_indices)

    classifier = model.classifier
    assert classifier.in_features % last_conv.out_channels == 0
    params_per_input_channel = classifier.in_features // last_conv.out_channels

    linear_pruned_indices = []
    for i in pruned_indices:
        linear_pruned_indices += list(range(i * params_per_input_channel, (i + 1) * params_per_input_channel))

    prune_linear_in_features_(classifier, linear_pruned_indices)
    # prune the output of the classifier
    prune_output_linear_layer_(classifier, group_indices, use_bce=args.bce)


def prune_resnet(model, candidates, group_indices):
    layers = model.children()
    # layer[0] : Conv2d
    # layer[1] : BatchNorm2e
    # layer[2] : ReLU
    layer_index = 3
    for stage in (layers[3], layers[4], layers[5]):
        for block in stage.children():
            assert isinstance(block, Bottleneck), "only support bottleneck block"
            children_dict = block.named_children()
            conv1 = children_dict['conv1']
            conv2 = children_dict['conv2']
            conv3 = children_dict['conv3']

            prune_contiguous_conv2d_(
                conv1, conv2, group_indices[layer_index], bn=children_dict['bn1'])
            layer_index += 1
            prune_contiguous_conv2d_(
                conv2, conv3, group_indices[layer_index], bn=children_dict['bn2'])
            layer_index += 2
            if 'downsample' in children_dict:
                layer_index += 1
    prune_output_linear_layer_(model.fc, group_indices, use_bce=args.bce)


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
            prune_vgg(new_model, candidates, group_indices)
        elif args.arch.startswith('resnet'):
            prune_resnet(new_model, candidates, group_indices)
        else:
            raise NotImplementedError

        # save the pruned model
        pruned_model_name = f"{args.arch}_({group_id})_pruned_model.pth"
        torch.save(new_model, os.path.join(
            model_save_directory, pruned_model_name))
        print('Pruned model saved at', model_save_directory)


if __name__ == '__main__':
    main()
