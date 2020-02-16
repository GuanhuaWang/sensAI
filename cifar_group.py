"""Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
"""
from __future__ import print_function
import json
import argparse
import os
import shutil
import time
import random
print("cifar_group.py is running.....'")
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
from datasets import cifar
import tqdm

from utils import Logger, AverageMeter, accuracy, accuracy_binary, mkdir_p, savefig

import re
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', required=True, type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('--group-id', required=True, type=int, help="The index of the group used by the model.")
parser.add_argument('--grouping-result-file', required=True, type=str, help="The path of the grouping result file.")
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8,
                    help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4,
                    help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12,
                    help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2,
                    help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--class-index', default=0, type=int,
                    help='class index for binary cls')
parser.add_argument('--pruned', default=False,
                    action='store_true', help='whether testing pruned models')
parser.add_argument('--bce', default=False, action='store_true',
                    help='Use binary cross entropy loss')
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy
model_prefix = 'retrain' # None


def main():
    print("cifar_group.py is running .......")
    print(args.checkpoint)
    global best_acc
    # Data
    print('==> Preparing dataset %s' % args.dataset)

    assert args.pruned

    with open(args.grouping_result_file) as f:
      grouping_result = json.load(f)
    class_indices = grouping_result[args.group_id]

    if args.dataset == 'cifar10':
        trainset = cifar.CIFAR10TrainingSetWrapper(class_indices, True)
        testset = cifar.CIFAR10TestingSetWrapper(class_indices, True)
    elif args.dataset == 'cifar100':
        trainset = cifar.CIFAR100TrainingSetWrapper(class_indices, True)
        testset = cifar.CIFAR100TestingSetWrapper(class_indices, True)
    else:
        raise NotImplementedError(f"There's no support for '{args.dataset}' dataset.")

    trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.train_batch,
            num_workers=args.workers,
            pin_memory=False)
    testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=args.test_batch,
            num_workers=args.workers,
            pin_memory=False)
    num_classes = len(class_indices) + 1

    if not args.pruned:
        # Model
        print("==> creating model '{}'".format(args.arch))
        if args.arch.startswith('resnext'):
            model = models.__dict__[args.arch](
                cardinality=args.cardinality,
                num_classes=num_classes,
                depth=args.depth,
                widen_factor=args.widen_factor,
                dropRate=args.drop,
            )
        elif args.arch.startswith('densenet'):
            model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=args.depth,
                growthRate=args.growthRate,
                compressionRate=args.compressionRate,
                dropRate=args.drop,
            )
        elif args.arch.startswith('wrn'):
            model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=args.depth,
                widen_factor=args.widen_factor,
                dropRate=args.drop,
            )
        elif args.arch.endswith('resnet'):
            model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=args.depth,
                block_name=args.block_name,
            )
        # else:
            # model = models.__dict__[args.arch](num_classes=num_classes)
    else:
        print("==> Loading pruned model with some existing weights '{}'".format(args.arch))
        model = torch.load(args.resume)
        if use_cuda:
            model.cuda()

    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    #if not os.path.isdir(args.checkpoint):  ##Kenan: Comment out these two lines, it is the reason we have folder everytime.
      #  mkdir_p(args.checkpoint)

    if use_cuda:
        model.cuda()
    # model = model.cpu()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel()
                                           for p in model.parameters())/1000000.0))

    if args.bce:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume and not args.pruned:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(
            args.resume), 'Error: no checkpoint directory found!'
       # args.checkpoint = os.path.dirname(args.resume)               ### Kenan: This line makes checkpoint dir as well, so we comment out.
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(model_prefix + '_log.txt', title=title, resume=True)
    else:
        logger = Logger(model_prefix + '_log.txt', title=title)
        logger.set_names(['Learning Rate', 'Train Loss',
                          'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if args.evaluate:
        print("args.evaluate is True")
        print('\nEvaluation only')
        test_loss, test_acc = test(
            testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    #print("We are saving here...")
    #print(str(args.epochs))
    if args.epochs != 0:      ###Kenan: Add a if statement to solve the error when epochs is 0
        for epoch in range(start_epoch, args.epochs):  
            adjust_learning_rate(optimizer, epoch)

            print('\nEpoch: [%d | %d] LR: %f' %
                  (epoch + 1, args.epochs, state['lr']))

            train_loss, train_acc = train(
                trainloader, model, criterion, optimizer, epoch, use_cuda)
            test_loss, test_acc = test(
                testloader, model, criterion, epoch, use_cuda)

            print('\nEpoch: [%d | %d] LR: %f  Test Acc: %f     Train Acc %f' % (
                epoch + 1, args.epochs, state['lr'], test_acc, train_acc))

            # append logger file
            logger.append([state['lr'], train_loss,
                           test_loss, train_acc, test_acc])

            # save model
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            #print("is_best being checked")
            if is_best:
                torch.save(model, args.checkpoint)
    else:
        torch.save(model, args.checkpoint)
    logger.close()
    logger.plot()
    savefig(model_prefix + '_log.eps')

    print('Bcst acc:')
    print(best_acc)


def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    with tqdm.tqdm(total=len(trainloader)) as bar:
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            bar.update()
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            outputs = model(inputs)

            if args.bce:
                loss = 0.0
                for i in range(outputs.shape[1]):
                    out = (outputs[:, i].flatten())
                    targ = targets.eq(i+1).float()
                    loss += criterion(outputs[:, i].flatten(),
                                    targets.eq(i+1).float())
            else:
                loss = criterion(outputs, targets)

            # measure accuracy and record loss
            if not args.bce:
                prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 2))
            else:
                prec1 = max([accuracy_binary(outputs.data[:, i], targets.data)
                            for i in range(outputs.shape[1])])
            losses.update(loss.item(), inputs.size(0))
            if not args.bce:
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))
            else:
                top1.update(prec1, inputs.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.set_description('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(trainloader),
                data=data_time.avg,
                bt=batch_time.avg,
                total='N/A' or bar.elapsed_td,
                eta='N/A' or bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            ))
    return (losses.avg, top1.avg)


def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with tqdm.tqdm(total=len(testloader)) as bar:
        for batch_idx, (inputs, targets) in enumerate(testloader):
            bar.update()
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                # compute output
                outputs = model(inputs)
                if args.bce:
                    # print('outputs', outputs[0])
                    # print('target', targets[0])
                    loss = 0.0
                    for i in range(outputs.shape[1]):
                        loss += criterion(outputs[:, i].flatten(),
                                        targets.eq(i+1).float())
                #    loss = criterion(outputs.flatten(), targets.float())
                else:
                    loss = criterion(outputs, targets)

                # measure accuracy and record loss
                if not args.bce:
                    prec1, prec5 = accuracy(
                        outputs.data, targets.data, topk=(1, 2))
                else:
                    prec1 = max([accuracy_binary(outputs.data[:, i], targets.data)
                                for i in range(outputs.shape[1])])
            losses.update(loss.item(), inputs.size(0))
            if not args.bce:
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))
            else:
                top1.update(prec1, inputs.size(0))

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

    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, model_prefix + '_model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    print("cifar10_group.py is running .....")
    main()
