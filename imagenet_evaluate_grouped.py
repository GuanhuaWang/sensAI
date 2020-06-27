import argparse
import os
import shutil
import time
import sys
import glob
import re
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import imagenet_dataset as datasets
import torchvision.models as models

from compute_flops import print_model_param_flops

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    args.evaluate = True

    cudnn.benchmark = True
    model_list = []
    num_flops = []
    avg_num_param = 0.0
    args.checkpoint = os.path.dirname(args.retrained_dir)
    criterion = nn.CrossEntropyLoss()

    # load groups
    file_names = [f for f in glob.glob(args.retrained_dir + "/" + args.arch + "/*.pth", recursive=False)]
    group_id_list = [filename_to_index(filename) for filename in file_names]
    group_config = np.load(open(args.retrained_dir + '/grouping_config.npy', "rb"))

    permutation_indices = []   # To allow for arbitrary grouping
    for group_id in group_id_list:
        permutation_indices.extend(group_config[int(group_id[0])])
    permutation_indices = torch.eye(1000)[permutation_indices].cuda(args.gpu)
    
    # load models
    for index, (group_id, file_name) in enumerate(zip(group_id_list, file_names)):
        model = torch.load(file_name)
        model = model.cuda(index % ngpus_per_node)
        avg_num_param += sum(p.numel() for p in model.parameters())/1000000.0
        print('Group {} model has total params: {:2f}M'.format(group_id ,sum(p.numel() for p in model.parameters())/1000000.0))
        model_list.append(model)
 
    # generate dataloader
    valdir = os.path.join(args.data, 'val')
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
   
    if args.evaluate:
        validate(val_loader, model_list, criterion, args, permutation_indices, ngpus_per_node)
        return

def validate(val_loader, model_list, criterion, args, p_indices, gpu_nums):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    for model in model_list:
        model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input_list = []
            for index in range(gpu_nums):
                input = input.cuda(index)
                input_list.append(input)
                target = target.cuda(0)     ### send same input and target to each gpu

            # compute output
            output_list = torch.Tensor().cuda(0)
            for index, model in enumerate(model_list):
                temp = model(input_list[index%gpu_nums])
                output = nn.Softmax(dim=1)(temp)[:, 1:]
                output_list= torch.cat((output_list, output), 1)
            output = torch.mm(output_list, p_indices)
       
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)
        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def filename_to_index(filename):
    filename = [int(s) for s in filename.split('_') if s.isdigit()]
    return filename


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res