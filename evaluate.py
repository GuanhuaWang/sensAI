'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import copy
import time
import os

from tqdm import tqdm

import torch
from torch import nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from utils import AverageMeter, accuracy

import itertools

from compute_flops import print_model_param_flops


class GroupedModel(nn.Module):
    def __init__(self, model_list, group_info, use_cuda):
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

        print("\n===== Metrics for grouped model ==========================\n")

        for group_id, model in zip(self.group_info, self.model_list):
            n_params = sum(p.numel() for p in model.parameters()) / 10 ** 6
            num_params.append(n_params)
            print(f'Grouped model for Class {group_id} '
                  f'Total params: {n_params:2f}M')
            num_flops.append(print_model_param_flops(model, 32))

        print(f"Average number of flops: {sum(num_flops) / len(num_flops) / 10 ** 9 :3f} G")
        print(f"Average number of param: {sum(num_params) / len(num_params)} M")


def evaluate_models(dataset_name, models_dir, grouping_result, use_cuda, n_workers=1, batch_size=256):
    if dataset_name == 'cifar10':
        dataset_loader = datasets.CIFAR10
        num_classes = 10
    elif dataset_name == 'cifar100':
        dataset_loader = datasets.CIFAR100
        num_classes = 100
    else:
        raise NotImplementedError

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
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers)

    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()  # if not args.bce else nn.BCEWithLogitsLoss()

    # load pruned models
    n_models = len(grouping_result)
    model_list = [torch.load(os.path.join(models_dir, f'group_{index}.pth')) for index in range(n_models)]
    group_info = copy.deepcopy(grouping_result)
    model = GroupedModel(model_list, group_info, use_cuda)
    model.print_statistics()

    # start test
    test_list(testloader, model, criterion, num_classes, use_cuda)


def test_list(testloader, model, criterion, num_classes, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if use_cuda:
        model.cuda()
    model.eval()
    end = time.time()

    confusion_matrix = np.zeros((num_classes, num_classes))

    bar = tqdm(total=len(testloader))
    # pdb.set_trace()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        bar.update(1)
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
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.set_description(
            '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
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

    np.set_printoptions(precision=3, linewidth=96)

    print("\n===== Full Confusion Matrix ==================================\n")
    if confusion_matrix.shape[0] < 20:
        print(confusion_matrix)
    else:
        print("Warning: The original confusion matrix is too big to fit into the screen. "
              "Skip printing the matrix.")

    print("\n===== Inter-group Confusion Matrix ===========================\n")
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

    print("\n===== In-group Confusion Matrix ==============================\n")
    for group in model.group_info:
        print(f"group {group}")
        inter_group_matrix = confusion_matrix[group, :][:, group]
        inter_group_matrix /= inter_group_matrix.sum(axis=-1)[:, np.newaxis]
        print(inter_group_matrix)
    # return losses.avg, top1.avg
