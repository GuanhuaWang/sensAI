CUDA_VISIBLE_DEVICES=1 python cifar_binary.py -a vgg19_bn --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/vgg19_bn_0 --train-batch 64 --class-index 0 &
CUDA_VISIBLE_DEVICES=2 python cifar_binary.py -a vgg19_bn --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/vgg19_bn_1 --train-batch 64 --class-index 1 &
CUDA_VISIBLE_DEVICES=3 python cifar_binary.py -a vgg19_bn --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/vgg19_bn_2 --train-batch 64 --class-index 2 &
CUDA_VISIBLE_DEVICES=4 python cifar_binary.py -a vgg19_bn --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/vgg19_bn_3 --train-batch 64 --class-index 3 &
CUDA_VISIBLE_DEVICES=5 python cifar_binary.py -a vgg19_bn --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/vgg19_bn_4 --train-batch 64 --class-index 4 &
CUDA_VISIBLE_DEVICES=1 python cifar_binary.py -a vgg19_bn --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/vgg19_bn_5 --train-batch 64 --class-index 5 &
CUDA_VISIBLE_DEVICES=2 python cifar_binary.py -a vgg19_bn --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/vgg19_bn_6 --train-batch 64 --class-index 6 &
CUDA_VISIBLE_DEVICES=3 python cifar_binary.py -a vgg19_bn --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/vgg19_bn_7 --train-batch 64 --class-index 7 &
CUDA_VISIBLE_DEVICES=4 python cifar_binary.py -a vgg19_bn --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/vgg19_bn_8 --train-batch 64 --class-index 8 &
CUDA_VISIBLE_DEVICES=5 python cifar_binary.py -a vgg19_bn --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/vgg19_bn_9 --train-batch 64 --class-index 9 &
wait

