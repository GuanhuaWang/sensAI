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


def group_classes(n_groups, dataset_name, arch, pretrained_model, seed, use_cuda,
                  group_selection_algorithm='even_k_means',
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
