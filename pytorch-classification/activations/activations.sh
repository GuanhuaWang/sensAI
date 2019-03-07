for i in 0 1 2 3 4 5 6 7 8 9
do
    python3 evaluate.py -a vgg19_bn --resume /home/ubuntu/baseModel/pytorch-classification/checkpoints/vgg19_bn_50_epochs/checkpoint.pth.tar --evaluate --class-index ${i} -s lc_logs_train_50_epoch > observe_logs/observe_${i}.log
done
