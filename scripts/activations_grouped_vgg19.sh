MODEL=./vgg19bn-cifar100.pth.tar
rm -r prune_candidate_logs
mkdir prune_candidate_logs
python3 get_prune_candidates.py -a vgg19_bn -d cifar100 --resume $MODEL --grouped  1 2 13 32 46 51 62 77 91 93
python3 get_prune_candidates.py -a vgg19_bn -d cifar100 --resume $MODEL --grouped  20 23 24 29 30 58 69 72 73 95
python3 get_prune_candidates.py -a vgg19_bn -d cifar100 --resume $MODEL --grouped  33 47 49 52 56 59 66 67 76 96
python3 get_prune_candidates.py -a vgg19_bn -d cifar100 --resume $MODEL --grouped  5 11 31 37 38 39 64 75 84 97
python3 get_prune_candidates.py -a vgg19_bn -d cifar100 --resume $MODEL --grouped  16 21 28 41 48 81 86 87 94 99
