import argparse
import os
import time

import torch
import torch.backends.cudnn as cudnn
import models.cifar as models
import numpy as np
from utils import Bar, AverageMeter
import logging

import dataset
from apoz_policy import ActivationRecord
from datasets import cifar10

logger = logging.Logger(__name__)

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(
    description='PyTorch CIFAR10/100 Generate Class Specific Information')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', required=True, default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
# Miscs
parser.add_argument('--seed', type=int, default=42, help='manual seed')
parser.add_argument('--class-index', default=0, type=int,
                    help='class index for class specific activation')
parser.add_argument('--grouped', required=True, type=int, nargs='+', default=[],
                    help='Generate activations based on the these class indices')


args = parser.parse_args()
use_cuda = torch.cuda.is_available() and True

# Random seed
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

global_record = ActivationRecord()
assert args.grouped


def main():
    dataset = cifar10.CIFAR10TrainingSetWrapper(args.grouped, False)
    pruning_loader = torch.data.utils.DataLoader(
        dataset,
        batch_size=1000,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False)

    # cudnn.benchmark = True
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint found!'
    checkpoint = torch.load(args.resume)
    # config = checkpoint['cfg']
    # model = models.__dict__[args.arch](dataset=args.dataset, depth=19, cfg=config)
    model = models.__dict__[args.arch](num_classes=10)
    if use_cuda:
        model.cuda()
    if use_cuda:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    print('\nMake a test run to generate activations. \n Using training set.\n')
    collect_pruning_data(pruning_loader, model, use_cuda)
    candidates_by_layer = global_record.generate_pruned_candidates()
    with open(f"prune_candidate_logs/class_({'_'.join(str(n) for n in args.grouped)})_apoz_layer_thresholds.npy", "wb") as f:
        np.save(f, candidates_by_layer)
    print(candidates_by_layer)


def collect_pruning_data(pruning_loader, model, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    if use_cuda:
        model.cuda()
    global_record.apply_hook(model)

    # guanhua
    print('===EVAL===')

    end = time.time()
    bar = Bar('Processing', max=len(pruning_loader))
    for batch_idx, (inputs, _) in enumerate(pruning_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs = inputs.cuda()

        with global_record.record():
            # we do not neet the output
            _ = model(inputs)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = (f'({batch_idx + 1}/{len(pruning_loader)}) Data: {data_time.avg:.3f}s | '
                      f'Batch: {batch_time.avg:.3f}s | Total: {bar.elapsed_td:} | ETA: {bar.eta_td:}')
        bar.next()
    bar.finish()


if __name__ == '__main__':
    main()
