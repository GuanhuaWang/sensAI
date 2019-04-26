EPOCHS=2
SAVE=./pruned2_retrain/vgg/
rm $SAVE -r
mkdir $SAVE
for file in ~/brandon/sensai_experiments/prune2/pruned2_models/vgg/*
do
    CUDA_VISIBLE_DEVICES=0 python3 cifar_group.py -a vgg --epochs ${EPOCHS} --pruned --schedule 21 30 --gamma 0.1 --resume $file --checkpoint $SAVE --train-batch 64

done
