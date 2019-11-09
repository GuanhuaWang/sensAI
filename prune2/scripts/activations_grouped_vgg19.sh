MODEL=./checkpoint_bearclaw.pth.tar
rm feature_map_data -r
rm prune_candidate_logs -r
mkdir feature_map_data
mkdir prune_candidate_logs
python3 activations.py -a vgg19_bn --resume $MODEL --grouped  0 7
python3 activations.py -a vgg19_bn --resume $MODEL --grouped  1 2
python3 activations.py -a vgg19_bn --resume $MODEL --grouped  3 4
python3 activations.py -a vgg19_bn --resume $MODEL --grouped  5 6
python3 activations.py -a vgg19_bn --resume $MODEL --grouped  8 9
