#!/bin/sh
EPOCHS=20
FROM=./pruned_models_30M_param
SAVE=${FROM}_retrained
mkdir ${SAVE}
rm ${SAVE}/* -r 
mkdir $SAVE/vgg19_bn/
mkdir $SAVE/logs/
i=0
group_idx=0
for file in ${FROM}/vgg19_bn/* 
    do
        # CUDA_VISIBLE_DEVICES=$((i))/ # ,$((i+1)),$((i+2)),$((i+3)),$((i+4)),$((i+5)),$((i+6)),$((i+7))\
        python3 imagenet_official_retrain.py /home/ubuntu/imagenet \
       --arch vgg19_bn \
       --epochs ${EPOCHS} \
       --config ${FROM}/grouping_config.npy \
       --schedule 10 15 \
       --gamma 0.1 \
       --resume $file \
       --checkpoint $SAVE/ &  # > ${SAVE}/logs/log${group_idx}.txt &
        
        # group_idx=$((group_idx+1))
        # i=$((i+1))
        # i=$(($i % 2))
        # if [ $i -eq 0 ];  then
        #     wait
        # fi
        wait
done

