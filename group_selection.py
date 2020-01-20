import argparse

import torch
import load_model
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

from even_k_means import kmeans_lloyd


def kmeans_grouping(features, targets, n_groups, same_group_size=True, seed=42):
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
        random_state=seed,
        tol=1e-6)
    groups = []
    for i in range(n_groups):
        groups.append(class_indices[best_labels == i].tolist())
    return groups


def group_classes(n_groups, dataset_name, arch, pretrained_model, seed, use_cuda, group_selection_algorithm='even_k_means',
                  n_workers=1, batch_size=256):
    print('==> Preparing dataset %s' % dataset_name)
    if dataset_name == 'cifar10':
        dataset_loader = datasets.CIFAR10
    elif dataset_name == 'cifar100':
        dataset_loader = datasets.CIFAR100
    else:
        raise NotImplementedError(
            f"There's no support for '{dataset_name}' dataset.")

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
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=False)

    model = load_model.load_pretrain_model(
        arch, dataset_name, pretrained_model, len(dataset.classes), use_cuda)

    all_features = []
    all_targets = []

    model.eval()
    print('\nMake a test run to generate groups. Using training set.\n')
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

    if group_selection_algorithm == 'even_k_means':
        groups = kmeans_grouping(all_features, all_targets,
                                 n_groups, same_group_size=True, seed=seed)
    else:
        raise ValueError(f"Unknown algorithm '{group_selection_algorithm}'")
    return groups


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch CIFAR10/100 Generate Group Info')
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
    parser.add_argument('-n', '--ngroups', required=True, type=int, metavar='N',
                        help='number of groups')

    # Miscs
    parser.add_argument('--seed', type=int, default=42, help='manual seed')

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available() and True

    # Random seed
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    groups = group_classes(args.ngroups,
                           arch=args.arch,
                           dataset=args.dataset,
                           pretrained_model=args.resume,
                           use_cuda=use_cuda,
                           seed=args.seed,
                           n_workers=args.workers)

    print("\n====================== Grouping Result ========================\n")
    for i, group in enumerate(groups):
        print(f"Group #{i}: {' '.join(str(idx) for idx in group)}")
