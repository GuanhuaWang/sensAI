for i in 0 1 2 3 4 5 6 7 8 9
do
    python3 evaluate.py -a vgg19_bn --resume checkpoints/cifar10/vgg19_bn/checkpoint.pth.tar --evaluate --class-index ${i} > observe_logs/observe_${i}.log
done
