mkdir exe_logs
mkdir squash_logs
mkdir feature_map_data
for i in 0 1 2 3 4 5 6 7 8 9
do
    python3 evaluate.py -a vgg19_bn --resume ~/baseModel/pytorch-classification/checkpoints/cifar10/vgg19_bn/checkpoint.pth.tar --evaluate --class-index ${i} --save-data feature_map_data > exe_logs/exe_class_${i}.log
done

