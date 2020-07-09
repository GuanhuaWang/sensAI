import re
import glob
import models.cifar as models
import os
import sys
import argparse
import pathlib
import pickle
import copy
import numpy as np 
import re 
import torch
from torch import nn
import load_model
import torch.multiprocessing as mp

from regularize_model import standard
from prune_utils.prune import prune_vgg16_conv_layer, prune_last_fc_layers, prune_resnet50
from prune_utils.layer_prune import (
    prune_output_linear_layer_,
    prune_contiguous_conv2d_,
    prune_conv2d_out_channels_,
    prune_batchnorm2d_,
    prune_linear_in_features_,
    prune_mobile_block,
    prune_shuffle_layer)
from models.cifar.resnet import Bottleneck
import torchvision.models as imagenet_models

parser = argparse.ArgumentParser(description='VGG with mask layer on cifar10')
parser.add_argument('-d', '--dataset', required=True, type=str)
parser.add_argument('-c', '--prune-candidates', default="./prune_candidate_logs/",
                    type=str, help='Directory which stores the prune candidates for each model')
parser.add_argument('-a', '--arch', default='vgg19_bn',
                    type=str, help='The architecture of the trained model')
parser.add_argument('-r', '--resume', default='', type=str,
                    help='The path to the checkpoints')
parser.add_argument('-s', '--save', default='./pruned_models',
                    type=str, help='The path to store the pruned models')
parser.add_argument('--bce', default=False, type=bool,
                    help='Prune according to binary cross entropy loss, i.e. no additional negative output for classifer')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
args = parser.parse_args()


def prune_vgg(model, pruned_candidates, group_indices):
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
    prune_output_linear_layer_(classifier, group_indices, use_bce=args.bce)


def prune_resnet(model, candidates, group_indices):
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
    prune_output_linear_layer_(model.fc, group_indices, use_bce=args.bce)

def prune_mobilenetv2(model, candidates, group_indices):
    layers = list(model.layers)
    layer_index = 1
    for block in layers:
        conv1 = block.conv1
        bn1 = block.bn1
        conv2 = block.conv2
        bn2 = block.bn2
        conv3 = block.conv3
        prune_1 = candidates[layer_index]
        prune_2 = candidates[layer_index+1]
        prune_mobile_block(conv1, conv2, conv3, prune_1, prune_2, bn1, bn2)
        layer_index += 2
    prune_output_linear_layer_(model.linear, group_indices, use_bce=args.bce)

def prune_shufflenetv2(model, candidates, group_indices):
    layer1, layer2, layer3 = model.layer1, model.layer2, model.layer3
    layer1_candidates = candidates[1:15]
    layer2_candidates = candidates[15:41]
    layer3_candidates = candidates[41:55]
    prune_shuffle_layer(layer1, layer1_candidates)
    prune_shuffle_layer(layer2, layer2_candidates)
    prune_shuffle_layer(layer3, layer3_candidates)
    prune_output_linear_layer_(model.linear, group_indices, use_bce=args.bce)

def filename_to_index(filename):
        filename = filename[6+len(args.prune_candidates):]
        return int(filename[:filename.index('_')])

def update_list(l):
    for i in range(len(l)):
        l[i] -= 1

def prune_cifar_worker(proc_ind, i, new_model, candidates, group_indices, arch, model_save_directory):
    num_gpus = torch.cuda.device_count()
    new_model.cuda(i % num_gpus)
    group_indices = group_indices.tolist()
    if args.arch.startswith('vgg'):
        prune_vgg(new_model, candidates, group_indices)
    elif args.arch.startswith('resnet'):
        prune_resnet(new_model, candidates, group_indices)
    elif args.arch.startswith('mobile'):
        prune_mobilenetv2(new_model, candidates, group_indices)
    elif args.arch.startswith('shuffle'):
        prune_shufflenetv2(new_model, candidates, group_indices)
    else:
        raise NotImplementedError

    # save the pruned model
    pruned_model_name = f"{arch}_{i}_pruned_model.pth"
    torch.save(new_model, os.path.join(
        model_save_directory, pruned_model_name))
    print('Pruned model saved at', model_save_directory)

def prune_imagenet_worker(proc_ind, model, candidates, group_indices, group_id, model_save_directory):
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(group_id % num_gpus)
    model.cuda(group_id % num_gpus)
    if args.arch != "resnet50":
        conv_indices = [idx for idx, (n, p) in enumerate(model.features._modules.items()) if isinstance(p, nn.modules.conv.Conv2d)]
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
        # The input dimension of the first fc layer is pruned from above
        model = prune_last_fc_layers(model, \
                                    group_indices, \
                                    filter_indices = candidates[offset:], \
                                    use_bce = args.bce)
    else:
        prune_resnet50(model, candidates, group_indices)

    pruned_model_name = args.arch + '_{}'.format(group_id) + '_pruned_model.pth'
    print('Grouped mode  %s Total params: %.2fM' % (group_id ,sum(p.numel() for p in model.parameters())/1000000.0))
    torch.save(model, os.path.join(model_save_directory, pruned_model_name))
    print('Pruned model saved at', model_save_directory)

def main():
    use_cuda = torch.cuda.is_available()
    # load groups
    file_names = [f for f in glob.glob(args.prune_candidates + "group_*.npy", recursive=False)]
    file_names.sort(key=filename_to_index)
    groups = np.load(open(args.prune_candidates + "grouping_config.npy", "rb"))
    
    # create pruned model save path
    model_save_directory = os.path.join(args.save, args.arch)
    pathlib.Path(model_save_directory).mkdir(parents=True, exist_ok=True)
    np.save(open(os.path.join(args.save, "grouping_config.npy"), "wb"), groups)
    if len(groups[0]) == 1:
        args.bce = True
    print(f'==> Preparing dataset {args.dataset}')
    if args.dataset in ['cifar10', 'cifar100']:
        if args.dataset == 'cifar10':
            num_classes = 10
        elif args.dataset == 'cifar100':
            num_classes = 100
        
        processes = []
        # for each class
        for i, (group_indices, file_name) in enumerate(zip(groups, file_names)):
            # load pruning candidates
            with open(file_name, 'rb') as f:
                candidates = pickle.load(f)
            # load checkpoints
            model = load_model.load_pretrain_model(
                args.arch, args.dataset, args.resume, num_classes, use_cuda)
            new_model = copy.deepcopy(model)
            if args.arch in ["mobilenetv2", "shufflenetv2"]:
                new_model = standard(new_model, args.arch, num_classes)
            p = mp.spawn(prune_cifar_worker, args=(i, new_model, candidates, group_indices, args.arch, model_save_directory), join=False)
            processes.append(p)
        for p in processes:
            p.join()
            

    elif args.dataset == 'imagenet':
        num_classes = len(groups)
        processes = []
        # for each class
        for group_id, file_name in enumerate(file_names):
            print('Pruning classes {} from candidates in {}'.format(group_id, file_name)) 
            group_indices = groups[group_id]
            # load pruning candidates
            print(file_name)
            candidates =  np.load(open(file_name, 'rb'), allow_pickle=True).tolist()
            
            num_gpus = torch.cuda.device_count()
            # load checkpoints
            if args.pretrained:
                print("=> using pre-trained model '{}'".format(args.arch))
                model = imagenet_models.__dict__[args.arch](pretrained=True)
                # model = torch.nn.DataParallel(model).cuda() #TODO use DataParallel
                model = model.cuda(group_id % num_gpus)
            else:
                checkpoint = torch.load(args.resume)
                model = imagenet_models.__dict__[args.arch](num_classes=num_classes)
                # model = torch.nn.DataParallel(model).cuda() #TODO use DataParallel
                model = model.cuda(group_id % num_gpus)
                model.load_state_dict(checkpoint['state_dict'])

            # join existing num_gpus processes, to make sure only num_gpus processes are running at a time
            if group_id % num_gpus == 0:
                for p in processes:
                    p.join()
                processes = []

            # model = model.module #TODO use DataParallel
            p = mp.spawn(prune_imagenet_worker, args=(model, candidates, group_indices, group_id, model_save_directory), join=False)
            processes.append(p)

        for p in processes:
            p.join()
    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()
