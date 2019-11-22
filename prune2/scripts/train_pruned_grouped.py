from training_scheduler import train
import os
import shutil
'''
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
'''

num_epochs = 80
model_dir = "./pruned_models"
save_dir = model_dir + "_retrained"
if os.path.isdir(save_dir):
    shutil.rmtree(save_dir)
os.mkdir(save_dir)
os.mkdir(save_dir+"/vgg19_bn/")
os.mkdir(save_dir+"/logs/")

i = 0
group_idx = 0
commands = []
for file in os.listdir(model_dir+"/vgg19_bn/"):
    command = "python3 cifar_group.py -a vgg19_bn --epochs " + str(num_epochs) +    " --pruned --schedule 40 60 --gamma 0.1 --resume " + model_dir + "/vgg19_bn/" + file + " --checkpoint " + save_dir + "/ --train-batch 256 --dataset cifar100 > " + save_dir + "/logs/log" + str(group_idx) + ".txt"
    group_idx += 1
    i = (i + 1) % 4
    commands.append(command)
print(commands)
# train(executables=commands)