rm feature_map_data -r
mkdir feature_map_data
for i in 0 1 2 3 4 5 6 7 8 9
do
    python3 activations.py -a vgg --resume checkpoint_network.pth.tar --evaluate --class-index ${i} 
done
