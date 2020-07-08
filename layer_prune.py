import torch
from torch import nn
from fractions import gcd

def prune_output_linear_layer_(linear_layer, class_indices, use_bce=False):
    if use_bce:
        assert len(class_indices) == 1
    else:
        # use 0 as the placeholder of the negative class
        class_indices = [0] + list(class_indices)
    linear_layer.bias.data = linear_layer.bias.data[class_indices]
    linear_layer.weight.data = linear_layer.weight.data[class_indices, :]
    if not use_bce:
        # reinitialize the negative sample class
        linear_layer.weight.data[0].normal_(0, 0.01)
    linear_layer.out_features = len(class_indices)


def prune_linear_in_features(fc, pruned_indices):
    new_fc = nn.Linear(fc.in_features - len(pruned_indices), fc.out_features)
    new_fc.bias.data = fc.bias.data.clone()
    new_fc.weight.data = prune_tensor(fc.weight.data, 1, pruned_indices)
    return new_fc


def prune_linear_in_features_(fc, pruned_indices):
    fc.in_features -= len(pruned_indices)
    fc.weight.data = prune_tensor(fc.weight.data, 1, pruned_indices)


def prune_tensor(tensor, dim, pruned_indices):
    if tensor.shape[dim] == 1:
        return tensor
    included_indices = [i for i in range(
        tensor.shape[dim]) if i not in pruned_indices]
    indexer = []
    for i in range(tensor.ndim):
        indexer.append(slice(None) if i != dim else included_indices)
    return tensor[indexer]

def prune_batchnorm2d(bn, pruned_indices):
    new_bn = nn.BatchNorm2d(bn.num_features - len(pruned_indices))
    new_bn.weight.data = prune_tensor(bn.weight.data, 0, pruned_indices)
    new_bn.bias.data = prune_tensor(bn.bias.data, 0, pruned_indices)
    new_bn.running_mean.data = prune_tensor(
        bn.running_mean.data, 0, pruned_indices)
    new_bn.running_var.data = prune_tensor(
        bn.running_var.data, 0, pruned_indices)
    return new_bn


def prune_batchnorm2d_(bn, pruned_indices):
    bn.num_features -= len(pruned_indices)
    bn.weight.data = prune_tensor(bn.weight.data, 0, pruned_indices)
    bn.bias.data = prune_tensor(bn.bias.data, 0, pruned_indices)
    bn.running_mean.data = prune_tensor(
        bn.running_mean.data, 0, pruned_indices)
    bn.running_var.data = prune_tensor(bn.running_var.data, 0, pruned_indices)
    return bn


def prune_conv2d_out_channels(conv, pruned_indices):
    new_conv = nn.Conv2d(in_channels=conv.in_channels,
                         out_channels=conv.out_channels - len(pruned_indices),
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         padding=conv.padding,
                         dilation=conv.dilation,
                         groups=conv.groups,
                         bias=conv.bias is not None)

    new_conv.weight.data = prune_tensor(conv.weight.data, 0, pruned_indices)

    if conv.bias is not None:
        new_conv.bias.data = prune_tensor(conv.bias.data, 0, pruned_indices)
    return new_conv


def prune_conv2d_out_channels_(conv, pruned_indices):
    conv.out_channels -= len(pruned_indices)
    conv.weight.data = prune_tensor(conv.weight.data, 0, pruned_indices)
    if conv.bias is not None:
        conv.bias.data = prune_tensor(conv.bias.data, 0, pruned_indices)
    return conv


def prune_conv2d_in_channels(conv, pruned_indices):
    new_conv = nn.Conv2d(in_channels=conv.in_channels - len(pruned_indices),
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         padding=conv.padding,
                         dilation=conv.dilation,
                         groups=conv.groups,
                         bias=conv.bias is not None)

    new_conv.weight.data = prune_tensor(conv.weight.data, 1, pruned_indices)

    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data.clone()
    return new_conv


def prune_conv2d_in_channels_(conv, pruned_indices):
    conv.in_channels -= len(pruned_indices)
    conv.weight.data = prune_tensor(conv.weight.data, 1, pruned_indices)
    return conv


def prune_contiguous_conv2d_(conv_p, conv_n, pruned_indices, bn=None):
    prune_conv2d_out_channels_(conv_p, pruned_indices)
    prune_conv2d_in_channels_(conv_n, pruned_indices)
    if bn is not None:
        prune_batchnorm2d_(bn, pruned_indices)

def prune_contiguous_conv2d_last(conv_p, conv_n, pruned_indices, bn=None):
    prune_conv2d_out_channels_(conv_p, pruned_indices)
    if bn is not None:
        prune_batchnorm2d_(bn, pruned_indices)

def prune_mobile_conv2d_in_channels(conv, pruned_indices):
    conv.in_channels -= len(pruned_indices)
    conv.groups = conv.in_channels

    conv.weight.data = prune_tensor(conv.weight.data, 1, pruned_indices)
    return conv

def prune_mobile_conv2d_out_channels(conv, pruned_indices):
    if conv.groups != 1:
      pruned_indices = pruned_indices[:(conv.out_channels - conv.in_channels)]
    conv.out_channels -= len(pruned_indices)
    conv.groups = conv.out_channels
    conv.weight.data = prune_tensor(conv.weight.data, 0, pruned_indices)
    return conv

def prune_contiguous_conv2d_mobile_a(conv_p, conv_n, pruned_indices, bn=None):
    prune_conv2d_out_channels_(conv_p, pruned_indices)
    prune_mobile_conv2d_in_channels(conv_n, pruned_indices)
    if bn is not None:
        prune_batchnorm2d_(bn, pruned_indices)

def prune_contiguous_conv2d_mobile_b(conv_p, conv_n, pruned_indices, bn=None):
    prune_mobile_conv2d_out_channels(conv_p, pruned_indices)
    prune_conv2d_in_channels_(conv_n, pruned_indices)
    if bn is not None:
        prune_batchnorm2d_(bn, pruned_indices[:conv_p.out_channels])

def prune_mobile_block(conv_1, conv_2, conv_3, pruned_indices_1, pruned_indices_2, bn_1, bn_2):
    small_len = min(len(pruned_indices_1), len(pruned_indices_2))
    if len(pruned_indices_2) < len(pruned_indices_1):
      pruned_indices_1 = pruned_indices_1[:small_len]
    prune_contiguous_conv2d_mobile_a(conv_1, conv_2, pruned_indices_1, bn=bn_1)
    prune_contiguous_conv2d_mobile_b(conv_2, conv_3, pruned_indices_2, bn=bn_2)

def prune_downblock(block, layer_candidates):
    conv3 = block.conv3
    bn3 = block.bn3
    conv4 = block.conv4
    bn4 = block.bn4
    conv5 = block.conv5
    pruned_indices_3_4 = layer_candidates[2]
    pruned_indices_4_5 = layer_candidates[3]
    small_len = min(len(pruned_indices_3_4), len(pruned_indices_4_5))
    if len(pruned_indices_4_5) < len(pruned_indices_3_4):
      pruned_indices_3_4 = pruned_indices_4_5[:small_len]
    prune_contiguous_conv2d_mobile_a(conv3, conv4, pruned_indices_3_4, bn=bn3)
    prune_contiguous_conv2d_mobile_b(conv4, conv5, pruned_indices_4_5, bn=bn4)

def prune_basicblock(block, layer_candidates):
    conv_1 = block.conv1 
    bn_1 = block.bn1 
    conv_2 = block.conv2
    bn_2 = block.bn2 
    conv_3 = block.conv3
    pruned_indices_1 = layer_candidates[0]
    pruned_indices_2 = layer_candidates[1]
    small_len = min(len(pruned_indices_1), len(pruned_indices_2))
    if len(pruned_indices_2) < len(pruned_indices_1):
      pruned_indices_1 = pruned_indices_1[:small_len]
    prune_contiguous_conv2d_mobile_a(conv_1, conv_2, pruned_indices_1, bn=bn_1)
    prune_contiguous_conv2d_mobile_b(conv_2, conv_3, pruned_indices_2, bn=bn_2)

def prune_shuffle_layer(layer, layer_candidates):
    for idx, block in enumerate(layer):
        if idx == 0:
            prune_downblock(block, layer_candidates[:5]) 
        else:
            candidates = layer_candidates[idx*3+2:idx*3+5]
            prune_basicblock(block, candidates)