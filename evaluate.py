import argparse
import os
import shutil
import time
import random
import warnings

from tqdm import tqdm
import torch
from torch import nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from utils import Logger, AverageMeter, accuracy, savefig
from torch.utils.data import Dataset, DataLoader
import glob
import re
import itertools
from compute_flops import print_model_param_flops
import torchvision.models as models
from imagenet_evaluate_grouped import main_worker
import torch.multiprocessing as mp

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
model_names += ["resnet110", "resnet164"]

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100/ImageNet Testing')
# Checkpoints
parser.add_argument('--retrained_dir', type=str, metavar='PATH',
                    help='path to the directory of pruned models (default: none)')
# Datasets
parser.add_argument('-d', '--dataset', required=True, type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--data', metavar='DIR', required=False,
                    help='path to imagenet dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--bce', default=False, action='store_true',
                    help='Use binary cross entropy loss')
best_acc1 = 0

# Miscs
parser.add_argument('--seed', type=int, default=42, help='manual seed')
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'imagenet', 'Dataset can only be cifar10, cifar100 or imagenet.'

# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed_all(args.seed)

torch.set_printoptions(threshold=10000)

def main():
    # imagenet evaluation
    if args.dataset == 'imagenet':
        imagenet_evaluate()
        return
    
    # cifar 10/100 evaluation
    print('==> Preparing dataset %s' % args.dataset)
    if args.dataset == 'cifar10':
        dataset_loader = datasets.CIFAR10
    elif args.dataset == 'cifar100':
        dataset_loader = datasets.CIFAR100
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
        batch_size = args.test_batch,
        shuffle = True,
        num_workers = args.workers)

    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    model = load_pruned_models(args.retrained_dir+'/'+args.arch+'/')

    if len(model.group_info) == 10 and args.dataset == 'cifar10':
        args.bce = True

    test_acc = test_list(testloader, model, criterion, use_cuda)

def imagenet_evaluate():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

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

    if args.dataset == 'cifar10':
        confusion_matrix = np.zeros((10, 10))
    elif args.dataset == 'cifar100':
        confusion_matrix = np.zeros((100, 100))
    else:
        raise NotImplementedError

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

    np.set_printoptions(precision=3, linewidth=96)

    print("\n===== Full Confusion Matrix ==================================\n")
    if confusion_matrix.shape[0] < 20:
        print(confusion_matrix)
    else:
        print("Warning: The original confusion matrix is too big to fit into the screen. "
              "Skip printing the matrix.")

    if all([len(group) > 1 for group in model.group_info]):
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
    return (losses.avg, top1.avg)

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
        if args.bce:
            for model_idx, model in enumerate(self.model_list):
                output = model(inputs)[:, 0]
                output_list.append(output)
            output_list = torch.softmax(torch.stack(output_list, dim=1).squeeze(), dim=1)
        else:
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
            n_params = sum(p.numel() for p in model.parameters()) / 10**6
            num_params.append(n_params)
            print(f'Grouped model for Class {group_id} '
                  f'Total params: {n_params:2f}M')
            num_flops.append(print_model_param_flops(model, 32))

        print(f"Average number of flops: {sum(num_flops) / len(num_flops) / 10**9 :3f} G")
        print(f"Average number of param: {sum(num_params) / len(num_params)} M")


def load_pruned_models(model_dir):
    group_dir = model_dir[:-(len(args.arch)+1)]
    if not model_dir.endswith('/'):
        model_dir += '/'
    file_names = [f for f in glob.glob(model_dir + "*.pth", recursive=False)]
    model_list = [torch.load(file_name, map_location=lambda storage, loc: storage.cuda(0)) for file_name in file_names]
    groups = np.load(open(group_dir + "grouping_config.npy", "rb"))
    group_info = []
    for file in file_names:
        group_id = filename_to_index(file)
        print(f"Group number is: {group_id}")
        class_indices = groups[group_id]
        group_info.append(class_indices.tolist()[0])
    model = GroupedModel(model_list, group_info)
    model.print_statistics()
    return model


def filename_to_index(filename):
    filename = [int(s) for s in filename.split('_') if s.isdigit()]
    return filename

if __name__ == '__main__':
    main()


