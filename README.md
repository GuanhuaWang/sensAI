# sensAI_experiments

## neuron prunning for VGG19_bn

To prune (dir: ~/zihao/rona_experiments/limo/vgg-pruning/)

`python3 prune_and_get_model.py -a vgg19_bn -r ~/baseModel/pytorch-classification/checkpoints/cifar10/vgg19_bn -c ~/guanhua/sensai_experiments/sensai/vgg-pruning/prune_candidate_logs/10per -s ./pruned_models_10per`

To evaluate binary classifiers (dir: ~/zihao/rona_experiments/pytorch-classification/)

`python3 evaluate.py -a vgg19_bn --test-batch 100 --resume ~/zihao/rona_experiments/limo/vgg-pruning/pruned_models_90per/vgg19_bn/ --evaluate --binary --pruned`
