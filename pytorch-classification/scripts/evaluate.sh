# python evaluate.py -a vgg19_bn --resume checkpoints/cifar10/vgg19_bn/checkpoint.pth.tar --evaluate
python3 evaluate.py -a vgg19_bn --test-batch 100 --resume ~/baseModel/pytorch-classification/checkpoints/cifar10/vgg19_bn/checkpoint.pth.tar --evaluate --binary
