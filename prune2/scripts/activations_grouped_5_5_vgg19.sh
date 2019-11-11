MODEL=./checkpoint_bearclaw.pth.tar
rm -r prune_candidate_logs
mkdir prune_candidate_logs

python3 activations.py -a vgg19_bn --resume $MODEL --evaluate --grouped  1 3 5 7 9
python3 activations.py -a vgg19_bn --resume $MODEL --evaluate --grouped  2 4 6 8 0
