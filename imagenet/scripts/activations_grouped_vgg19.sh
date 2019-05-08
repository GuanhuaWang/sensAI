MODEL=./checkpoint_bearclaw.pth.tar
rm feature_map_data -r
mkdir feature_map_data
python3 activations.py -a vgg19_bn --resume $MODEL --evaluate --grouped  0 1
python3 activations.py -a vgg19_bn --resume $MODEL --evaluate --grouped  2 3
python3 activations.py -a vgg19_bn --resume $MODEL --evaluate --grouped  4 5
python3 activations.py -a vgg19_bn --resume $MODEL --evaluate --grouped  6 7
python3 activations.py -a vgg19_bn --resume $MODEL --evaluate --grouped  8 9
# python3 activations.py -a vgg --resume $MODEL --evaluate --grouped  5
#python3 activations.py -a vgg --resume $MODEL --evaluate --grouped  6
# python3 activations.py -a vgg --resume $MODEL --evaluate --grouped  7 
#python3 activations.py -a vgg --resume $MODEL --evaluate --grouped  8 
# python3 activations.py -a vgg --resume $MODEL --evaluate --grouped  9 
