# sensAI_experiments

## neuron prunning for VGG19_bn

To prune

`python3 prune_and_get_model.py -a vgg19_bn -r ~/baseModel/pytorch-classification/checkpoints/cifar10/vgg19_bn -c ~/guanhua/sensai_experiments/sensai/vgg-pruning/prune_candidate_logs/10per -s ./pruned_models_10per`
