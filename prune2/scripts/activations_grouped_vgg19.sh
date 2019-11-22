MODEL=./checkpoint_bearclaw.pth.tar
rm -r prune_candidate_logs
mkdir prune_candidate_logs
python3 get_prune_candidates.py -a vgg19_bn --resume $MODEL --grouped  0 7
python3 get_prune_candidates.py -a vgg19_bn --resume $MODEL --grouped  1 2
python3 get_prune_candidates.py -a vgg19_bn --resume $MODEL --grouped  3 4
python3 get_prune_candidates.py -a vgg19_bn --resume $MODEL --grouped  5 6
python3 get_prune_candidates.py -a vgg19_bn --resume $MODEL --grouped  8 9
