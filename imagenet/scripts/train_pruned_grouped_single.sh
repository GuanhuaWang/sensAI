EPOCHS=160
SAVE=./pruned2_grouped_retrain_10M_160epoch/vgg/
i=0
FILE=~/brandon/sensai_experiments/prune2/pruned2_models_grouped_10M/vgg/vgg_'('8_9')'_pruned_model.pth
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 cifar_group.py -a vgg --epochs ${EPOCHS} --pruned --schedule 80 120 --gamma 0.1 --resume ${FILE} --checkpoint $SAVE --train-batch 64 &

