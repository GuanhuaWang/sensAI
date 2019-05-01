MODEL=./checkpoint_network.pth.tar
rm feature_map_data -r
mkdir feature_map_data
python3 activations.py -a vgg --resume $MODEL --evaluate --grouped  1 3 5 7 9
python3 activations.py -a vgg --resume $MODEL --evaluate --grouped  2 4 6 8 0
# python3 activations.py -a vgg --resume $MODEL --evaluate --grouped  4 5
# python3 activations.py -a vgg --resume $MODEL --evaluate --grouped  6 7
# python3 activations.py -a vgg --resume $MODEL --evaluate --grouped  8 9
# python3 activations.py -a vgg --resume $MODEL --evaluate --grouped  5
#python3 activations.py -a vgg --resume $MODEL --evaluate --grouped  6
# python3 activations.py -a vgg --resume $MODEL --evaluate --grouped  7 
#python3 activations.py -a vgg --resume $MODEL --evaluate --grouped  8 
# python3 activations.py -a vgg --resume $MODEL --evaluate --grouped  9 
