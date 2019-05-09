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

### Training more than one model at once on a single machine 
During imagenet retrain process we wanted to train 4 models at once across 16 gpus, to fully utilize each gpu.
After exec 4 training process, each with 4 CUDA_VISIBLE_GPUs,
Each program essentially grinds to a halt. 
It takes 5 minutes to arrive at training begin (model loading / data set loading)
And it takes 100s of seconds to score a single test batch of size 128. 
It might be guessed that when training 4 models on one machine, we're bounded by the CPUs ability
to load the data between GPU and CPU (data loading workers). 

Suprisingly, when executing the same style of parallism during the activation observation / channel scoring
procedure, we ARE able to launch 10 models across 10 GPUs. 

The main differences I observed between these two situations are: 
Training (Forward + Backwards) vs Inference (Forward only)
Multiple GPUS vs Single GPUS (Though I did try to launch just two models at once each with a single GPU)
In training situation, zombie threads (O_O) from killed processes, still living and eating up CPU cycles .
Worth looking into from a system perspective? If for some reason people want to train multiple models at once on a single machine.
