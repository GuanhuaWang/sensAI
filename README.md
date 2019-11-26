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


1. Generate groups by running:

   ```bash
   python3 group_selection.py --arch=$ARCH --resume=vgg19bn-cifar100.pth.tar --dataset=$DATASET --ngroups=10
   ```

2. Specify pruning parameters in `apoz_policy.py`

3. Specify group indices in `./scripts/activations_grouped_vgg19.sh`

4. To generate pruning candidates run `./scripts/activations_grouped_vgg19.sh`

5. Pruning candidate now stored in `./prune_candidate_logs`

6. Generate pruned model, 
   
   First, delete `pruned_models/` dir if existed.

   Then, run

   ```bash
   python3 prune_and_get_model.py -a $ARCH --dataset $DATASET --resume ./checkpoint_bearclaw.pth.tar  -c ./prune_candidate_logs/ -s ./TO_SAVE_MODEL_BASE_DIR
   ```

   Models now saved at location, `./TO_SAVE_MODEL_BASE_DIR`

7. Specify from dir (pruned models) and save dir (pruned and retrained models), and training parameters in `./scripts/train_pruned_grouped.sh`

   After above script runs, retrained model located at specified save dir.

8. To evaluate,

   ```bash
   python3 evaluate.py --dataset=$DATASET ./PATH_TO_RETRAIN_SAVED_DIR/ --test-batch 128
   ```
