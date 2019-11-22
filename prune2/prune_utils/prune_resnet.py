import torch
from torchvision import models
import numpy as np

import torch.nn as nn
from models import *

from prune_utils.layer_prune import (
    prune_tensor, prune_batchnorm2d_,
    prune_conv2d_out_channels_,
    prune_conv2d_in_channels_)


def prune_resnet_conv_layer_(model, layer_index, pruned_indices, use_batch_norm=False):
    layers = list(model.modules())
    conv = layers[layer_index]
    next_conv = None

    for i in range(layer_index + 1, len(layers)):
        res = layers[layer_index + i]
        if isinstance(res, nn.Conv2d):
            next_conv = res
            break

    prune_conv2d_out_channels_(conv, pruned_indices)

    if use_batch_norm:
        bn = layers[layer_index + 1]
        assert isinstance(bn, nn.BatchNorm2d)
        prune_batchnorm2d_(bn, pruned_indices)

    if next_conv is not None:
        prune_conv2d_in_channels_(next_conv, pruned_indices)


def prune_first_conv_layer_(model, layer_index, pruned_indices):
    conv = list(model.modules())[layer_index]
    # Change number of input channels for conv layer
    prune_conv2d_in_channels_(conv, pruned_indices)


def prune_selection_layer(model, layer_index, filters_to_remove):
    layers = list(model.modules())
    # Find previous channel selection layer
    offset = 1

    chan_sel = None
    while layer_index - offset > 0:
        res = layers[layer_index-offset]
        if isinstance(res, channel_selection):
            chan_sel = res
            break
        offset += 1

    # Can only prune input channels if there is selection layer before it
    if chan_sel is None:
        print("No selection layer")
        return model

    # Set channel selection layer to add zero mask
    chan_sel.indexes.data[filters_to_remove] = 0.0
    return model
