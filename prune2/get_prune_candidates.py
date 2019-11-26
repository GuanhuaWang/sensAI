import argparse
import pickle

import torch
from torch import nn
import torch.backends.cudnn as cudnn

from apoz_policy import ActivationRecord
from datasets import cifar
import load_model
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description='PyTorch CIFAR10/100 Generate Class Specific Information')
# Datasets
parser.add_argument('-d', '--dataset', required=True, type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', required=True, default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=load_model.model_arches('cifar'),
                    help='model architecture: ' +
                    ' | '.join(load_model.model_arches('cifar')) +
                    ' (default: resnet18)')
# Miscs
parser.add_argument('--seed', type=int, default=42, help='manual seed')
parser.add_argument('--grouped', required=True, type=int, nargs='+', default=[],
                    help='Generate activations based on the these class indices')


args = parser.parse_args()
use_cuda = torch.cuda.is_available() and True

# Random seed
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed_all(args.seed)

assert args.grouped


def main():
    if args.dataset == 'cifar10':
        dataset = cifar.CIFAR10TrainingSetWrapper(args.grouped, False)
        num_classes = 10
    elif args.dataset == 'cifar100':
        dataset = cifar.CIFAR100TrainingSetWrapper(args.grouped, False)
        num_classes = 100
    else:
        raise NotImplementedError(
            f"There's no support for '{args.dataset}' dataset.")

    pruning_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1000,
        num_workers=args.workers,
        pin_memory=False)

    model = load_model.load_pretrain_model(
        args.arch, 'cifar', args.resume, num_classes, use_cuda)

    print('\nMake a test run to generate activations. \n Using training set.\n')
    with ActivationRecord(model) as recorder:
        # collect pruning data
        bar = tqdm(total=len(pruning_loader))
        for batch_idx, (inputs, _) in enumerate(pruning_loader):
            bar.update(1)
            if use_cuda:
                inputs = inputs.cuda()
            recorder.record_batch(inputs)
        candidates_by_layer = recorder.generate_pruned_candidates()

    with open(f"prune_candidate_logs/class_({'_'.join(str(n) for n in args.grouped)})_apoz_layer_thresholds.npy", "wb") as f:
        pickle.dump(candidates_by_layer, f)
    print(candidates_by_layer)


if __name__ == '__main__':
    main()
