MODEL=/home/ubuntu/baseModel/pytorch-classification/checkpoints/cifar100/resnet-164/model_best.pth.tar
rm -r prune_candidate_logs
mkdir prune_candidate_logs
python3 get_prune_candidates.py -a resnet164 -d cifar100 --resume $MODEL --grouped  0 10 53 54 57 61 62 70 83 92
python3 get_prune_candidates.py -a resnet164 -d cifar100 --resume $MODEL --grouped  28 30 39 67 69 71 73 91 95 99
python3 get_prune_candidates.py -a resnet164 -d cifar100 --resume $MODEL --grouped  5 9 16 20 22 25 84 86 87 94
python3 get_prune_candidates.py -a resnet164 -d cifar100 --resume $MODEL --grouped  34 35 36 38 50 65 66 88 97 98
python3 get_prune_candidates.py -a resnet164 -d cifar100 --resume $MODEL --grouped  6 7 14 15 19 24 40 51 75 79
python3 get_prune_candidates.py -a resnet164 -d cifar100 --resume $MODEL --grouped  23 33 47 49 52 56 59 60 82 96
python3 get_prune_candidates.py -a resnet164 -d cifar100 --resume $MODEL --grouped  18 26 27 29 42 44 74 77 78 93
python3 get_prune_candidates.py -a resnet164 -d cifar100 --resume $MODEL --grouped  2 8 11 41 45 46 48 58 85 89
python3 get_prune_candidates.py -a resnet164 -d cifar100 --resume $MODEL --grouped  3 4 21 31 43 55 63 64 72 80
python3 get_prune_candidates.py -a resnet164 -d cifar100 --resume $MODEL --grouped  1 12 13 17 32 37 68 76 81 90
