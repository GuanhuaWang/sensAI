#!/bin/sh
EPOCHS=160
SAVE=./pruned2_grouped_retrain_10M_160epoch_bce/
rm ${SAVE}
mkdir $SAVE
mkdir $SAVE/vgg/
mkdir $SAVE/logs/
i=0
group_idx=0
for file in ~/brandon/sensai_experiments/prune2/pruned2_models_grouped_10M_bce/vgg/* 
    do
        CUDA_VISIBLE_DEVICES=$i python3 cifar_group.py -a vgg --epochs ${EPOCHS} --pruned --bce --schedule 80 120 --gamma 0.1 --resume $file --checkpoint $SAVE --train-batch 64 > ${SAVE}/logs/log${group_idx}.txt &
        group_idx=$((group_idx+1))
        i=$((i+1))
        i=$(( $i  % 4 ))
        if [ $i -eq 0 ] ;  then
            wait
        fi
done

