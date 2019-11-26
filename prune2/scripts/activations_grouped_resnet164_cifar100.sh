MODEL=/home/ubuntu/baseModel/pytorch-classification/checkpoints/cifar100/resnet-164/model_best.pth.tar
rm -r prune_candidate_logs
mkdir prune_candidate_logs
python3 get_prune_candidates.py -a resnet164 -d cifar100 --resume $MODEL --grouped  1 2 13 32 46 51 62 77 91 93
python3 get_prune_candidates.py -a resnet164 -d cifar100 --resume $MODEL --grouped  20 23 24 29 30 58 69 72 73 95
python3 get_prune_candidates.py -a resnet164 -d cifar100 --resume $MODEL --grouped  33 47 49 52 56 59 66 67 76 96
python3 get_prune_candidates.py -a resnet164 -d cifar100 --resume $MODEL --grouped  5 11 31 37 38 39 64 75 84 97
python3 get_prune_candidates.py -a resnet164 -d cifar100 --resume $MODEL --grouped  16 21 28 41 48 81 86 87 94 99
python3 get_prune_candidates.py -a resnet164 -d cifar100 --resume $MODEL --grouped  12 15 17 25 60 68 71 85 89 90
python3 get_prune_candidates.py -a resnet164 -d cifar100 --resume $MODEL --grouped  3 6 19 34 35 36 43 65 80 88
python3 get_prune_candidates.py -a resnet164 -d cifar100 --resume $MODEL --grouped  0 9 14 54 57 63 82 83 92 98
python3 get_prune_candidates.py -a resnet164 -d cifar100 --resume $MODEL --grouped  8 10 22 26 40 50 53 61 70 79
python3 get_prune_candidates.py -a resnet164 -d cifar100 --resume $MODEL --grouped  4 7 18 27 42 44 45 55 74 78
