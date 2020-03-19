import os

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


def prune_vgg(model, pruned_candidates, group_indices, use_bce=False):
    features = model.features
    conv_indices = [i for i, layer in enumerate(features) if isinstance(layer, nn.Conv2d)]

    conv_bn_indices = [i for i, layer in enumerate(features) if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d))]
    assert len(conv_indices) == len(pruned_candidates)
    assert len(conv_indices) * 2 == len(conv_bn_indices)

    for i, conv_index in enumerate(conv_indices[:-1]):
        next_conv = None
        for j in range(conv_index + 1, len(features)):
            l = features[j]
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
    classifier = model.classifier
    assert classifier.in_features % last_conv.out_channels == 0
    params_per_input_channel = classifier.in_features // last_conv.out_channels

    pruned_indices = pruned_candidates[-1]
    prune_conv2d_out_channels_(last_conv, pruned_indices)
    prune_batchnorm2d_(features[conv_bn_indices[-1]], pruned_indices)

    linear_pruned_indices = []
    for i in pruned_indices:
        linear_pruned_indices += list(range(i * params_per_input_channel, (i + 1) * params_per_input_channel))

    prune_linear_in_features_(classifier, linear_pruned_indices)
    # prune the output of the classifier
    prune_output_linear_layer_(classifier, group_indices, use_bce=use_bce)


def prune_resnet(model, candidates, group_indices, use_bce=False):
    layers = list(model.children())
    # layer[0] : Conv2d
    # layer[1] : BatchNorm2e
    # layer[2] : ReLU
    layer_index = 1
    for stage in (layers[3], layers[4], layers[5]):
        for block in stage.children():
            assert isinstance(block, Bottleneck), "only support bottleneck block"
            children_dict = dict(block.named_children())
            conv1 = children_dict['conv1']
            conv2 = children_dict['conv2']
            conv3 = children_dict['conv3']

            prune_contiguous_conv2d_(
                conv1, conv2, candidates[layer_index], bn=children_dict['bn1'])
            layer_index += 1
            prune_contiguous_conv2d_(
                conv2, conv3, candidates[layer_index], bn=children_dict['bn2'])
            layer_index += 2
            # because we are using the output of the ReLU, the output of
            # the downsample is merged before ReLU, so we do not need to
            # increase the layer index
    prune_output_linear_layer_(model.fc, group_indices, use_bce=use_bce)


def prune_model_from_pretrained(dataset_name, arch, pretrained_model, grouping_result, pruning_candidates,
                                model_saving_dir, use_cuda):
    if dataset_name == 'cifar10':
        num_classes = 10
    elif dataset_name == 'cifar100':
        num_classes = 100
    else:
        raise NotImplementedError

    for i, (group_indices, candidates) in enumerate(zip(grouping_result, pruning_candidates)):
        # load checkpoints
        model = load_model.load_pretrain_model(
            arch, dataset_name, pretrained_model, num_classes, use_cuda)
        new_model = copy.deepcopy(model)

        # pruning
        if arch.startswith('vgg'):
            prune_vgg(new_model, candidates, group_indices)
        elif arch.startswith('resnet'):
            prune_resnet(new_model, candidates, group_indices)
        else:
            raise NotImplementedError

        # save the pruned model
        pruned_model_name = f"group_{i}.pth"
        torch.save(new_model, os.path.join(model_saving_dir, pruned_model_name))
    print('Pruned model saved at', model_saving_dir)
