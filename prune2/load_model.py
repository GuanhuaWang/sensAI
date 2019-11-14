import os
import torch

import models.cifar as cifar_models


def model_arches(dataset):
    if dataset == 'cifar':
        return sorted(name for name in cifar_models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(cifar_models.__dict__[name]))
    else:
        raise NotImplementedError



def load_pretrain_model(arch, dataset, resume_checkpoint, num_classes, use_cuda):
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(resume_checkpoint), 'Error: no checkpoint found!'
    if use_cuda:
        checkpoint = torch.load(resume_checkpoint)
    else:
        checkpoint = torch.load(
            resume_checkpoint, map_location=torch.device('cpu'))
    # config = checkpoint['cfg']
    # model = models.__dict__[args.arch](dataset=args.dataset, depth=19, cfg=config)
    if dataset == 'cifar':
        model = cifar_models.__dict__[arch](num_classes=num_classes)
    else:
        raise NotImplementedError

    if use_cuda:
        model.cuda()
    state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
    model.load_state_dict(state_dict)
    return model
