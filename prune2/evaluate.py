'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time

from tqdm import tqdm

import torch
from torch import nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from utils import Bar, Logger, AverageMeter, accuracy, savefig
from torch.utils.data import Dataset, DataLoader

import glob
import re

import itertools

from compute_flops import print_model_param_flops

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Testing')
# Checkpoints
parser.add_argument('model_dir', type=str, metavar='PATH',
                    help='path to the directory of pruned models (default: none)')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                    help='test batchsize')

# Miscs
parser.add_argument('--seed', type=int, default=42, help='manual seed')
args = parser.parse_args()

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed_all(args.seed)

torch.set_printoptions(threshold=10000)


class GroupedModel(nn.Module):
    def __init__(self, model_list, group_info):
        super().__init__()
        self.group_info = group_info
        # flatten list of list
        permutation_indices = list(itertools.chain.from_iterable(group_info))
        self.permutation_indices = torch.eye(len(permutation_indices))[permutation_indices]
        if use_cuda:
            self.permutation_indices = self.permutation_indices.cuda()
        self.model_list = nn.ModuleList(model_list)

    def forward(self, inputs):
        output_list = []
        for model_idx, model in enumerate(self.model_list):
            output = torch.softmax(model(inputs), dim=1)[:, 1:]
            output_list.append(output)
        output_list = torch.cat(output_list, 1)
        return torch.mm(output_list, self.permutation_indices)

    def print_statistics(self):
        num_params = []
        num_flops = []

        for group_id, model in zip(self.group_info, self.model_list):
            n_params = sum(p.numel() for p in model.parameters()) / 10**6
            num_params.append(n_params)
            print(f'Grouped model for Class {group_id} '
                  f'Total params: {n_params:2f}M')
            num_flops.append(print_model_param_flops(model, 32))

        print(f"Average number of flops: {sum(num_flops) / len(num_flops) / 10**9 :3f} G")
        print(f"Average number of param: {sum(num_params) / len(num_params)} M")


def load_pruned_models(model_dir):
    if not model_dir.endswith('/'):
        model_dir += '/'
    file_names = [f for f in glob.glob(model_dir + "*.pth", recursive=False)]
    group_id_list = [re.search('\(([^)]+)', f_name).group(1)
                     for f_name in file_names]

    print(f"Grouping settings found: {group_id_list}")

    assert bool(file_names) and bool(
        group_id_list), "No files found. Maybe wrong directory?"

    model_list = [torch.load(file_name) for file_name in file_names]
    group_info = [[int(ind) for ind in g.split('_')] for g in group_id_list]
    model = GroupedModel(model_list, group_info)
    model.print_statistics()
    return model


def main():
    # Data
    print('==> Preparing dataset %s' % args.dataset)
    if args.dataset == 'cifar10':
        dataset_loader = datasets.CIFAR10
    else:
        dataset_loader = datasets.CIFAR100

    testloader = data.DataLoader(
        dataset_loader(
            root='./data',
            download=False,
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])),
        batch_size = args.test_batch,
        shuffle = True,
        num_workers = args.workers)

    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()  # if not args.bce else nn.BCEWithLogitsLoss()
    model = load_pruned_models(args.model_dir)

    test_acc = test_list(testloader, model, criterion, use_cuda)


def test_list(testloader, model, criterion, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if use_cuda:
        model.cuda()
    model.eval()
    end = time.time()

    confusion_matrix = np.zeros((10, 10))

    bar = tqdm(total=len(testloader))
    # pdb.set_trace()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        bar.update(batch_idx)
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            for output, target in zip(outputs, targets):
                gt = target.item()
                dt = np.argmax(output.cpu().numpy())
                confusion_matrix[gt, dt] += 1
            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk = (1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.set_description('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(testloader),
            data=data_time.avg,
            bt=batch_time.avg,
            total='N/A' or bar.elapsed_td,
            eta='N/A' or bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        ))
    bar.close()

    print("===== Full Confusion Matrix ===========================")
    if confusion_matrix.shape[0] < 20:
        print(confusion_matrix)
    else:
        print("Warning: The original confusion matrix is too big to fit into the screen. "
              "Skip printing the matrix.")

    print("===== Inter-group Confusion Matrix ===========================")
    print(f"Group info: {[group for group in model.group_info]}")
    n_groups = len(model.group_info)
    group_confusion_matrix = np.zeros((n_groups, n_groups))
    for i in range(n_groups):
        for j in range(n_groups):
            cols = model.group_info[i]
            rows = model.group_info[j]
            group_confusion_matrix[i, j] += confusion_matrix[cols[0], rows[0]]
            group_confusion_matrix[i, j] += confusion_matrix[cols[0], rows[1]]
            group_confusion_matrix[i, j] += confusion_matrix[cols[1], rows[0]]
            group_confusion_matrix[i, j] += confusion_matrix[cols[1], rows[1]]
    group_confusion_matrix /= group_confusion_matrix.sum(axis=-1)[:, np.newaxis]
    print(group_confusion_matrix)

    print("===== In-group Confusion Matrix ===========================")
    for group in model.group_info:
        print(f"group {group}")
        inter_group_matrix = confusion_matrix[group, :][:, group]
        inter_group_matrix /= inter_group_matrix.sum(axis=-1)[:, np.newaxis]
        print(inter_group_matrix)
    return (losses.avg, top1.avg)


if __name__ == '__main__':
    main()


