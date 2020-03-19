import torch
import load_model
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def extract_features(dataset_name, arch, pretrained_model, use_cuda,
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
    return all_features, all_targets
