MODEL=/home/ubuntu/baseModel/pytorch-classification/checkpoints/cifar100/resnet-110/model_best.pth.tar
rm -r prune_candidate_logs
mkdir prune_candidate_logs
python3 get_prune_candidates.py -a resnet110 -d cifar100 --resume $MODEL --grouped  0 10 53 54 57 62 70 82 83 92
python3 get_prune_candidates.py -a resnet110 -d cifar100 --resume $MODEL --grouped  23 30 32 49 61 67 71 73 91 95
python3 get_prune_candidates.py -a resnet110 -d cifar100 --resume $MODEL --grouped  5 16 20 25 28 40 84 86 87 94
python3 get_prune_candidates.py -a resnet110 -d cifar100 --resume $MODEL --grouped  15 19 34 38 42 43 66 75 88 97
python3 get_prune_candidates.py -a resnet110 -d cifar100 --resume $MODEL --grouped  9 11 12 17 37 39 68 69 76 98
python3 get_prune_candidates.py -a resnet110 -d cifar100 --resume $MODEL --grouped  18 26 27 29 44 45 78 79 93 99
python3 get_prune_candidates.py -a resnet110 -d cifar100 --resume $MODEL --grouped  8 13 41 46 48 58 81 85 89 90
python3 get_prune_candidates.py -a resnet110 -d cifar100 --resume $MODEL --grouped  14 22 33 47 51 52 56 59 60 96
python3 get_prune_candidates.py -a resnet110 -d cifar100 --resume $MODEL --grouped  3 4 21 31 55 63 64 72 74 80
python3 get_prune_candidates.py -a resnet110 -d cifar100 --resume $MODEL --grouped  1 2 6 7 24 35 36 50 65 77
