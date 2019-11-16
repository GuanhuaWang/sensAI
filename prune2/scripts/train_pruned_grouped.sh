#!/bin/sh
EPOCHS=80
FROM=./pruned_models
SAVE=${FROM}_retrained
mkdir ${SAVE}
rm ${SAVE}/* -r 
mkdir $SAVE/vgg19_bn/
mkdir $SAVE/logs/
i=0
group_idx=0
for file in ${FROM}/vgg19_bn/* 
    do
        CUDA_VISIBLE_DEVICES=$i python3 cifar_group.py -a vgg19_bn --epochs ${EPOCHS} --pruned --schedule 40 60 --gamma 0.1 --resume $file --checkpoint $SAVE/ --train-batch 256 --dataset cifar100 > ${SAVE}/logs/log${group_idx}.txt &
        group_idx=$((group_idx+1))
        i=$((i+1))
        i=$(( $i  % 4 ))
        if [ $i -eq 0 ] ;  then
            wait
        fi
done

