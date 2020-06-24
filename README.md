# sensAI experiment

## Pre-requirement

Linux, python 3.6+

## Setup

```bash
pip install -r requirements.txt
```

## Experiment Instructions

Supported neural network architectures for cifar10/cifar100: ARCH = {vgg19_bn, resnet110, resnet164}

Supported neural network architectures for imagenet: ARCH = {vgg19_bn, resnet50}


1. Generate groups by running:
   
   For cifar10/cifar100:
   ```bash
   python3 group_selection.py --arch $ARCH --resume $pretrained_model --dataset $DATASET --ngroups $number_of_groups --gpu_num $number_of_gpu 
   ```
   For imagenet:
   ```bash
   python3 group_selection.py --arch $ARCH --dataset imagenet --ngroups $number_of_groups --gpu_num $number_of_gpu --data /path_to_imagenet_dataset
   ```
   
   Pruning candidate now stored in ./prune_candidate_logs
   
2. Prune models:
    
   For cifar10/cifar100:
   ```bash
   python3 prune_and_get_model.py -a $ARCH --dataset $DATASET --resume $pretrained_model  -c ./prune_candidate_logs/ -s ./TO_SAVE_PRUNED_MODEL_DIR
   ```
   For imagenet:
   ```bash
   python3 prune_and_get_model.py -a $ARCH --dataset imagenet -c ./prune_candidate_logs/ -s ./TO_SAVE_PRUNED_MODEL_DIR --pretrained
   ```
   
   Pruned models are now saved in ./TO_SAVE_PRUNED_MODEL_DIR/$ARCH
   
3. Retrain pruned models:
  
   For cifar10/cifar100:
   ```bash
   python3 retrain_grouped_model.py -a $ARCH --dataset $DATASET --resume ./TO_SAVE_PRUNED_MODEL_DIR/ --train_batch $batch_size --epochs $number_you_choose --num_gpus $number_of_gpus_to_run_on
   ```
   For imagenet:
   ```bash
   python3 retrain_grouped_model.py -a $ARCH --dataset imagenet --resume ./TO_SAVE_PRUNED_MODEL_DIR/ --epochs $number_you_choose --num_gpus $number_of_gpus_to_run_on --train_batch $batch_size --data /path_to_imagenet_dataset
   ```
   
   Retrained models now saved in ./TO_SAVE_PRUNED_MODEL_DIR_retrained/$ARCH/
   
4. Evaluate

   For cifar10/cifar100:
   ```bash
   python3 evaluate.py -a $ARCH --dataset=$DATASET --retrained_dir ./TO_SAVE_PRUNED_MODEL_DIR_retrained --test-batch $batch_size
   ```
   For imagenet:
   ```bash
   python3 evaluate.py -d imagenet -a $ARCH --retrained_dir ./TO_SAVE_PRUNED_MODEL_DIR_retrained --data /path_to_imagenet_dataset
   ```