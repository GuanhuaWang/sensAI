#!/bin/sh
EPOCHS=2
FROM=./pruned_models
SAVE=${FROM}_retrained
mkdir ${SAVE}
rm ${SAVE}/* -r 
mkdir $SAVE/resnet164/
mkdir $SAVE/logs/
i=0
group_idx=0
for file in ${FROM}/resnet164/* 
    do
        CUDA_VISIBLE_DEVICES=$i python3 cifar_group.py -a resnet164 --epochs ${EPOCHS} --pruned --schedule 40 60 --gamma 0.1 --resume $file --checkpoint $SAVE/ --train-batch 256 --dataset cifar100 > ${SAVE}/logs/log${group_idx}.txt &
        group_idx=$((group_idx+1))
        i=$((i+1))
        i=$(( $i  % 4 ))
        if [ $i -eq 0 ] ;  then
            wait
        fi
done

