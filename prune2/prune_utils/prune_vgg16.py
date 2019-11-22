import torch
from torch import nn
from torchvision import models
import numpy as np

from prune_utils.layer_prune import (
    prune_tensor, prune_batchnorm2d,
    prune_conv2d_out_channels,
    prune_conv2d_in_channels,
    prune_linear_in_features)


def _vgg_get_last_linear_layer(model):
    layer_index = 0
    linear_layer = None
    if len(model.classifier._modules):
        for _, module in model.classifier._modules.items():
            if isinstance(module, nn.Linear):
                linear_layer = module
                break
            layer_index += 1
    else:
        linear_layer = model.classifier

    if linear_layer is None:
        raise ValueError("No linear layer found in classifier")
    return layer_index, linear_layer


def prune_vgg16_conv_layer(model, layer_index, pruned_indices, use_batch_norm=False):
    conv = model.features[layer_index]
    model.features[layer_index] = prune_conv2d_out_channels(conv, pruned_indices)

    if use_batch_norm:
        bn = model.features[layer_index + 1]
        assert isinstance(bn, nn.BatchNorm2d)
        model.features[layer_index + 1] = prune_batchnorm2d(bn, pruned_indices)

    next_conv = None
    offset = 1

    while layer_index + offset < len(model.features):
        _l = model.features[layer_index+offset]
        if isinstance(_l, nn.Conv2d):
            next_conv = _l
            break
        offset += 1

    if next_conv is not None:
        model.features[layer_index + offset] = prune_conv2d_in_channels(next_conv, pruned_indices)
    else:
        # Prunning the last conv layer. This affects the first linear layer of the classifier.
        layer_index, old_linear_layer = _vgg_get_last_linear_layer(model)
        assert old_linear_layer.in_features % conv.out_channels == 0
        params_per_input_channel = old_linear_layer.in_features // conv.out_channels

        linear_pruned_indices = []
        for i in pruned_indices:
            linear_pruned_indices += list(range(i * params_per_input_channel, (i + 1) * params_per_input_channel))

        new_linear_layer = prune_linear_in_features(old_linear_layer, linear_pruned_indices)
        if len(model.classifier._modules):
            model.classifier[layer_index] = new_linear_layer
        else:
            model.classifier = new_linear_layer

    return model


def prune_last_fc_layer(model, class_indices, use_bce=False):
    layer_index, old_linear_layer = _vgg_get_last_linear_layer(model)

    bce_offset = 0 if use_bce else 1
    # TODO: optimize this by reusing original indices

    new_linear_layer = nn.Linear(
        old_linear_layer.in_features, len(class_indices) + bce_offset)

    new_linear_layer.weight.data[bce_offset:] = old_linear_layer.weight.data[class_indices, :]
    new_linear_layer.bias.data[bce_offset:] = old_linear_layer.bias.data[class_indices]

    if len(model.classifier._modules):
        model.classifier[layer_index] = new_linear_layer
    else:
        model.classifier = new_linear_layer

    return model
