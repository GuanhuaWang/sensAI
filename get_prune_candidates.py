import torch

from apoz_policy import ActivationRecord
from datasets import cifar
import load_model
from tqdm import tqdm


def _get_candidates_of_classes(classes, dataset_name, arch, pretrained_model, use_cuda, n_workers=1, batch_size=256):
    if dataset_name == 'cifar10':
        dataset = cifar.CIFAR10TrainingSetWrapper(classes, False)
        num_classes = 10
    elif dataset_name == 'cifar100':
        dataset = cifar.CIFAR100TrainingSetWrapper(classes, False)
        num_classes = 100
    else:
        raise NotImplementedError(
            f"There's no support for '{dataset_name}' dataset.")

    model = load_model.load_pretrain_model(
        arch, dataset_name, pretrained_model, num_classes, use_cuda)

    pruning_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=False)

    with ActivationRecord(model) as recorder:
        # collect pruning data
        bar = tqdm(total=len(pruning_loader))
        for batch_idx, (inputs, _) in enumerate(pruning_loader):
            bar.update(1)
            if use_cuda:
                inputs = inputs.cuda()
            recorder.record_batch(inputs)
    candidates_by_layer = recorder.generate_pruned_candidates()
    return candidates_by_layer


def get_candidates_from_pretrained(
        dataset_name, arch, pretrained_model, grouping_result, use_cuda, n_workers=1, batch_size=256):
    print('\nMake a test run to generate activations. \n Using training set.\n')
    candidates_from_pretrained = []
    for classes in grouping_result:
        candidates_by_layer = _get_candidates_of_classes(classes, dataset_name, arch, pretrained_model, use_cuda,
                                                         n_workers=n_workers,
                                                         batch_size=batch_size)
        candidates_from_pretrained.append(candidates_by_layer)

    return candidates_from_pretrained
