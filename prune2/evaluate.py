'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
import pdb
import numpy as np
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from torch.utils.data import Dataset, DataLoader

import dataset
import glob
import re

from compute_flops import print_model_param_flops

class MyTestDataset(Dataset):
    def __init__(self):
        self.cifar10 = datasets.CIFAR10(root='./data',
                                        download=False,
                                        train=False,
                                        transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))
        
    def __getitem__(self, index):
        data, target = self.cifar10[index]
        
        # Your transformations here (or set it in CIFAR10)
        
        return data, target

    def __len__(self):
        return len(self.cifar10)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--test-batch', default=1000, type=int, metavar='N',
                    help='test batchsize')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--pruned', default=False, action='store_true', help='whether testing pruned models')
parser.add_argument('--binary', action='store_true', help='whether to use binary testing')
parser.add_argument('--prune-test', action='store_true', help='Set to true to prune according to test set ')
parser.add_argument('--refined', default=False, action='store_true', help='Set to true to retrain pruned model ')
parser.add_argument('--grouped', default=False, action='store_true', help='Set to true to evaluate for grouped models')
parser.add_argument('--bce', default=False, action='store_true', help='Using BCE rather than CE')
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy
num_classes = 2
torch.set_printoptions(threshold=10000)

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)



    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    if args.binary:
        num_classes = 2

    
    # trainset = MyTrainDataset()
    # trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = MyTestDataset()
    testloader = data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=args.workers)

    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss() # if not args.bce else nn.BCEWithLogitsLoss()

    if args.evaluate:
        if args.binary:
            model_list = []
            args.checkpoint = os.path.dirname(args.resume)
            if args.pruned:
                for i in range(10):
                    if not args.refined:
                        pruned_model_name = args.arch + '_{}'.format(i) + '_pruned_model.pth'
                    else:
                        pruned_model_name = args.arch + '_{}'.format(i) + '_pruned_model/model.pth' 
                    model_path = os.path.join(args.resume, pruned_model_name)
                    model = torch.load(model_path)
                    model = torch.nn.DataParallel(model).cuda()
                    print('Binary model for Class %i Total params: %.2fM' % (i ,sum(p.numel() for p in model.parameters())/1000000.0))
                    model_list.append(model)
            else:
                for i in range(10):
                    model = models.__dict__[args.arch](dataset=args.dataset, depth=16)
                    # model = torch.nn.DataParallel(model).cuda() 
                    folder = args.checkpoint + '_{}'.format(i)
                    checkpoint = torch.load(os.path.join(folder, 'checkpoint.pth.tar'))
                    model.load_state_dict(checkpoint['state_dict'])
                    print('Binary model for Class %i Total params: %.2fM' % (i ,sum(p.numel() for p in model.parameters())/1000000.0))
                    model_list.append(model)
            test_acc = test_list(testloader, model_list, criterion, start_epoch, use_cuda)
        if args.grouped:
            model_list = []
            num_flops = []
            avg_num_param = 0.0
            args.checkpoint = os.path.dirname(args.resume)
            print(args.resume)
            file_names = [f for f in glob.glob(args.resume + "*.pth", recursive=False)]
            print(file_names)
            group_id_list = [re.search('\(([^)]+)', f_name).group(1) for f_name in file_names]
            print(group_id_list)
            permutation_indices = [int(c) for c in "".join(group_id_list).replace("_","")]
            permutation_indices = torch.eye(10)[permutation_indices].cuda()
            for group_id, file_name in zip(group_id_list, file_names):
                # model = models.__dict__[args.arch](dataset=args.dataset, depth=16)
                model = torch.load(file_name)
                # model = torch.nn.DataParallel(model).cuda()
                avg_num_param += sum(p.numel() for p in model.parameters())/1000000.0
                print('Grouped model for Class {} Total params: {:2f}M'.format(group_id ,sum(p.numel() for p in model.parameters())/1000000.0))
                num_flops.append(print_model_param_flops(model, 32))
                model_list.append(model.cuda())
            print("Average number of flops: ", sum(num_flops) / float(len(num_flops)))
            print("Average number of param: ", avg_num_param / float(len(num_flops)))
            
            test_acc = test_list(testloader, model_list, criterion, start_epoch, use_cuda, permutation_indices)

"""
            if args.pruned:
                pass
            else:
                for 
"""         

def test_list(testloader, model_list, criterion, epoch, use_cuda, p_indices):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    # model.eval()
    for i in range(len(model_list)):
        model_list[i].eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    # pdb.set_trace()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        output_list = torch.Tensor().cuda()
        neg_list = []
        if args.pruned:
            for model_idx, model in enumerate(model_list):
                output_current = model(inputs)[:, 1].unsqueeze(1)
                output_list = torch.cat((output_list, output_current), 1)
        elif args.bce:
           for model_idx, model in enumerate(model_list):
                output = nn.Softmax(dim=1)(model(inputs))
                # print(output[0])
                # input()
                output_list = torch.cat((output_list, output), 1)
                # print(output_list)
                # prnt(input)
        else:
            for idx, model in enumerate(model_list):
                output = nn.Softmax(dim=1)(model(inputs))[:,1:]
                output_list = torch.cat((output_list, output), 1)
        output_list = torch.mm(output_list, p_indices)
            
        # print("")
        # print(targets[0], ['{:.2f}'.format(x) for x in output_list[0]] )
        # input()
            # print(neg_list[0])

  
        outputs = output_list
        loss = criterion(outputs, targets)
#        pdb.set_trace()
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
       
    bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
    bar.next()
     
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

if __name__ == '__main__':
    main()
