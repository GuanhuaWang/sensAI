import torch
from torch import nn


def prune_output_linear_layer_(linear_layer, class_indices, use_bce=False):
    if use_bce:
        assert len(class_indices) == 1
    else:
        # use 0 as the placeholder of the negative class
        class_indices = [0] + class_indices
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
