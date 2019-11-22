import torch
from torchvision import models
import numpy as np

import torch.nn as nn
from models import *


def prune_resnet_conv_layer(model, layer_index, filter_index, use_batch_norm=False):
    conv = list(model.modules())[layer_index]
    next_conv = None
    offset = 1

    while layer_index + offset < len(list(model.modules())):
        res = list(model.modules())[layer_index+offset]
        if isinstance(res, torch.nn.Conv2d):
            next_conv = res
            break
        offset = offset + 1

    new_conv = \
        torch.nn.Conv2d(in_channels=conv.in_channels,
                        out_channels=conv.out_channels - 1,
                        kernel_size=conv.kernel_size,
                        stride=conv.stride,
                        padding=conv.padding,
                        dilation=conv.dilation,
                        groups=conv.groups,
                        bias=True)  # conv.bias)

    old_weights = conv.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()
    new_weights[:filter_index, :, :, :] = old_weights[:filter_index, :, :, :]
    new_weights[filter_index:, :, :, :] = old_weights[filter_index + 1:, :, :, :]
    conv.weight.data = torch.from_numpy(new_weights).cuda()

    if conv.bias is not None:
        bias_numpy = conv.bias.data.cpu().numpy()

        bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
        bias[:filter_index] = bias_numpy[:filter_index]
        bias[filter_index:] = bias_numpy[filter_index + 1:]
        conv.bias.data = torch.from_numpy(bias).cuda()

    conv.out_channels -= 1

    if use_batch_norm:
        bn = list(model.modules())[layer_index + 1]
        new_bn = torch.nn.BatchNorm2d(conv.out_channels)

        old_weights = bn.weight.data.cpu().numpy()
        new_weights = new_bn.weight.data.cpu().numpy()
        new_weights[:filter_index] = old_weights[:filter_index]
        new_weights[filter_index:] = old_weights[filter_index+1:]

        old_bias = bn.bias.data.cpu().numpy()
        new_bias = new_bn.bias.data.cpu().numpy()
        new_bias[:filter_index] = old_bias[:filter_index]
        new_bias[filter_index:] = old_bias[filter_index+1:]

        old_running_mean = bn.running_mean.cpu().numpy()
        new_running_mean = new_bn.running_mean.cpu().numpy()
        new_running_mean[:filter_index] = old_running_mean[:filter_index]
        new_running_mean[filter_index:] = old_running_mean[filter_index+1:]

        old_running_var = bn.running_var.cpu().numpy()
        new_running_var = new_bn.running_var.cpu().numpy()
        new_running_var[:filter_index] = old_running_var[:filter_index]
        new_running_var[filter_index:] = old_running_var[filter_index+1:]

        bn.weight.data = torch.from_numpy(new_weights).cuda()
        bn.bias.data = torch.from_numpy(new_bias).cuda()
        bn.running_mean = torch.from_numpy(new_running_mean).cuda()
        bn.running_var = torch.from_numpy(new_running_var).cuda()

        bn.num_features -= 1

    if not next_conv is None:
        old_weights = next_conv.weight.data.cpu().numpy()
        old_shape = old_weights.shape
        new_weights = np.zeros(
            (old_shape[0], old_shape[1]-1, old_shape[2], old_shape[3]), dtype='f')
        new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
        new_weights[:, filter_index:, :, :] = old_weights[:, filter_index + 1:, :, :]
        next_conv.weight.data = torch.from_numpy(new_weights).cuda()

        next_conv.in_channels -= 1

    return model


def prune_first_conv_layer(model, layer_index, filter_index):
    conv = list(model.modules())[layer_index]

    # Change number of input channels for conv layer
    old_weights = conv.weight.data.cpu().numpy()
    old_shape = old_weights.shape
    new_weights = np.zeros(
        (old_shape[0], old_shape[1]-1, old_shape[2], old_shape[3]), dtype='f')
    new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
    new_weights[:, filter_index:, :, :] = old_weights[:, filter_index + 1:, :, :]
    conv.weight.data = torch.from_numpy(new_weights).cuda()
    # print(conv.weight.data.shape)

    conv.in_channels -= 1

    return model


def prune_selection_layer(model, layer_index, filters_to_remove):
    # Find previous channel selection layer
    chan_sel = None
    offset = 1

    while layer_index - offset > 0:
        res = list(model.modules())[layer_index-offset]
        if isinstance(res, channel_selection):
            chan_sel = res
            break
        offset = offset + 1

    # Can only prune input channels if there is selection layer before it
    if chan_sel is None:
        print("No selection layer")
        return model

    # Set channel selection layer to add zero mask
    chan_sel.indexes.data[filters_to_remove] = 0.0
    return model
