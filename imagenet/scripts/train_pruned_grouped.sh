#!/bin/sh
EPOCHS=10
FROM=./pruned_models_conservative
SAVE=${FROM}_retrained
mkdir ${SAVE}
rm ${SAVE}/* -r 
mkdir $SAVE/vgg19_bn/
mkdir $SAVE/logs/
i=0
group_idx=0
for file in ${FROM}/vgg19_bn/* 
    do
        python3 imagenet_official_retrain.py /home/ubuntu/imagenet --arch vgg19_bn --epochs ${EPOCHS} --config ${FROM}/grouping_config.npy \
                                           --schedule 40 60 --gamma 0.1 --resume $file --checkpoint $SAVE/ # > ${SAVE}/logs/log${group_idx}.txt &
        group_idx=$((group_idx+1))
        wait

done

