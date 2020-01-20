# sensAI experiment

## Pre-requirement

Linux, python 3.6+

## Setup

```bash
pip install -r requirements.txt
```

## Experiment Instructions

Supported neural network architectures: ARCH = {vgg19_bn, resnet110, resnet164}

Supported datasets: DATASET = {cifar10, cifar100}

1. Specify pruning parameters in `apoz_policy.py`

2. Creating a configure YAML file, and then execute it like:

   ```bash
   ./auto_script.py vgg19bn_cifar100_10groups.yaml
   ```
 
   Look into `outputs` for result files. Use `./auto_script.py --help` for more options.


NOTE: we haven't tested the retraining part