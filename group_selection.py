import argparse
import pickle

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import load_model
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import subprocess as sp
import os

from even_k_means import kmeans_lloyd

parser = argparse.ArgumentParser(
    description='PyTorch CIFAR10/100/Imagenet Generate Group Info')
# Datasets
parser.add_argument('-d', '--dataset', required=True, type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--data', default='/home/ubuntu/imagenet', required=False, type=str,
                    help='location of the imagenet dataset that includes train/val')
# Architecture
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet20',
                    #choices=load_model.model_arches('cifar'),
                    help='model architecture: ' +
                    ' | '.join(load_model.model_arches('cifar')) +
                    ' (default: resnet18)')
parser.add_argument('-n', '--ngroups', required=True, type=int, metavar='N',
                    help='number of groups')
parser.add_argument('-g', '--gpu_num', default=1, type=int, 
                    help='number of gpus')           

# Miscs
parser.add_argument('--seed', type=int, default=42, help='manual seed')
args = parser.parse_args()
use_cuda = torch.cuda.is_available() and True

# Random seed
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed_all(args.seed)

def main():
    print('==> Preparing dataset %s' % args.dataset)
    resultExist = os.path.exists("./prune_candidate_logs")
    if resultExist:
        rm_cmd = 'rm -rf ./prune_candidate_logs'
        sp.Popen(rm_cmd, shell=True)
    mkdir_cmd = 'mkdir ./prune_candidate_logs'
    sp.Popen(mkdir_cmd, shell=True)
    # cifar10/100 group selection
    if args.dataset in ['cifar10', 'cifar100']:
        if args.dataset == 'cifar10':
            dataset_loader = datasets.CIFAR10
        elif args.dataset == 'cifar100':
            dataset_loader = datasets.CIFAR100

        dataset = dataset_loader(
            root='./data',
            download=True,
            train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]))
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1000,
            num_workers=args.workers,
            pin_memory=False)

        model = load_model.load_pretrain_model(
            args.arch, 'cifar', args.resume, len(dataset.classes), use_cuda)

        all_features = []
        all_targets = []

        model.eval()
        print('\nMake a test run to generate groups. \n Using training set.\n')
        with tqdm(total=len(data_loader)) as bar:
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                bar.update()
                if use_cuda:
                    inputs = inputs.cuda()
                with torch.no_grad():
                    features = model(inputs, features_only=True)
                all_features.append(features)
                all_targets.append(targets)

        all_features = torch.cat(all_features)
        all_targets = torch.cat(all_targets)

        groups = kmeans_grouping(all_features, all_targets,
                                args.ngroups, same_group_size=True)
        print("groups: ", groups)
        print("\n====================== Grouping Result ========================\n")
        process_list = [None for _ in range(args.gpu_num)]
        for i, group in enumerate(groups):
            if process_list[i % args.gpu_num]:
                process_list[i % args.gpu_num].wait()
            print(f"Group #{i}: {' '.join(str(idx) for idx in group)}")
            exec_cmd = 'python3 get_prune_candidates.py' +\
                     ' -a %s' % args.arch + ' -d %s' % args.dataset + ' --resume ./%s' % args.resume + \
                     ' --grouped ' + str(group)[1:-1].replace(",", "") + ' --group_number %d' % i + ' --gpu_num %d' % (i % args.gpu_num)
            process_list[i % args.gpu_num]  = sp.Popen(exec_cmd, shell=True)
        
        np.save(open("prune_candidate_logs/grouping_config.npy", "wb"), groups)
        
    # imagenet group selection
    elif args.dataset == 'imagenet':
        num_gpus = args.gpu_num
        num_groups = args.ngroups
        group_size = 1000 // num_groups
        groups = [[i for i in range((j) * group_size, (j+1) * group_size)] for j in range(num_groups) ]
        process_list = [None for _ in range(num_gpus)]
        for i, group in enumerate(groups):
            if process_list[i % num_gpus]:
                process_list[i % num_gpus].wait()
            exec_cmd = 'python3 imagenet_activations.py ' +\
                     ' --data %s' % args.data +\
                     ' --gpu %d' % (i % num_gpus) +\
                     ' --arch %s' % args.arch + ' --evaluate --pretrained --group %s' % (' '.join(str(digit) for digit in group)) + \
                     ' --name %s' % (str(i))
            process_list[i % num_gpus]  = sp.Popen(exec_cmd, shell=True)
        # Save the grouping class index partition information
        np.save(open("prune_candidate_logs/grouping_config.npy", "wb"), groups)
    else:
        raise NotImplementedError(f"There's no support for '{args.dataset}' dataset.")

def kmeans_grouping(features, targets, n_groups, same_group_size=True):
    class_indices = targets.unique().sort().values
    mean_vectors = []
    for t in class_indices:
        mean_vec = features[targets == t.item(), :].mean(dim=0)
        mean_vectors.append(mean_vec.cpu().numpy())
    X = np.asarray(mean_vectors)
    class_indices = class_indices.cpu().numpy()
    assert X.ndim == 2
    best_labels, best_inertia, best_centers, _ = kmeans_lloyd(
        X, None, n_groups, verbose=True,
        same_cluster_size=same_group_size,
        random_state=args.seed,
        tol=1e-6)
    groups = []
    for i in range(n_groups):
        groups.append(class_indices[best_labels == i].tolist())
    return groups

if __name__ == '__main__':
    main()
