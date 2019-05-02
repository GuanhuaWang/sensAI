# sensAI_experiments

## neuron prunning for VGG19_bn

To prune (dir: ~/zihao/rona_experiments/limo/vgg-pruning/)

`python3 prune_and_get_model.py -a vgg19_bn -r ~/baseModel/pytorch-classification/checkpoints/cifar10/vgg19_bn -c ~/guanhua/sensai_experiments/sensai/vgg-pruning/prune_candidate_logs/10per -s ./pruned_models_10per`

To evaluate binary classifiers (dir: ~/zihao/rona_experiments/pytorch-classification/)

`python3 evaluate.py -a vgg19_bn --test-batch 100 --resume ~/zihao/rona_experiments/limo/vgg-pruning/pruned_models_90per/vgg19_bn/ --evaluate --binary --pruned`

## Further exploration

### Binary model grouping

One interesting question that might be an interesting,
Is whether the grouping matters.
Comparing random vs heuristic? (Using most similar / least similar, according to the heatmap statistics in the poster)

Random is more general. Heuristic is useful for reducing grouped model size (which I think we can do a follow-up paper with some system optimizations)

Might take time to collect enough sample points to make a discrimination.

### when to kill a binary retrain process

Some binary models converge faster than others, set threshold to kill converged models in order to save compute resources.

### DataFrame for irregular binary models
