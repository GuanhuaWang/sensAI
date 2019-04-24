import numpy as np
import os
import sys
import argparse
import pathlib
import glob
import re

path = './prune_candidate_logs/'
file_names = [f for f in glob.glob(path + "*.npy", recursive=False)]
print(file_names)
group_id_list = [re.search('\(([^)]+)', f_name).group(1) for f_name in file_names]
num_groups = len(group_id_list)

parser = argparse.ArgumentParser(description='Observe properties of feature maps')
parser.add_argument('-c', '--prune-candidates', default="./prune_candidate_logs", type=str, help='Directory which stores the prune candidates for each model')

args = parser.parse_args()

def main():
    class_candidates = []
    global_shared = [] # Each entry is a list of (layer_idx, channel_idx) elements for each class, 10 entries
    layer_shared = [[] for i in range(16)] 
    total_pruned = 0
    for group_id, file_name in zip(group_id_list, file_names):
        # load pruning candidates
        candidates = np.load(open(file_name, 'rb')).tolist()
        class_candidates.append(candidates)
        num_candidates_group_i =  sum(len(layer) for layer in candidates)
        total_pruned += num_candidates_group_i
        class_i_candidates_flat = []
        print("Num candidate for class {}: ".format(group_id), num_candidates_group_i)
        for layer_idx, pc_in_layer in enumerate(candidates):
            layer_shared[layer_idx].append(pc_in_layer)
            class_i_candidates_flat.extend([(layer_idx, c_idx) for c_idx in pc_in_layer])
        global_shared.append(class_i_candidates_flat) 
    
    avg_num_prune = total_pruned / (num_groups*1.0)
    # print(global_shared)
    set_list = [set(c) for c in global_shared] 
    # print(set_list) # Uncomment to see channels pruned for each class in a flattened list (layer_idx, channel_idx)
    global_shared = set.intersection(*set_list)
    # On average how many are pruned for each class. Then how many of those are shared with everyone
    shared_matrix = [[[] for _ in range(num_groups)] for _ in range(num_groups)]  # Each entry in the shared matrix contains the channels class i shares with class j
    print("Total shared globally out of all average {} / {}".format(len(global_shared), avg_num_prune)) 
    for idx, layer in enumerate(layer_shared):
         for class_idx in range(len(layer)):
             for class_jdx in range(len(layer)):
                 if class_idx != class_jdx:
                     shared_ij_layer_k = list(set.intersection(set(layer[class_idx]), set(layer[class_jdx])))
                     shared_matrix[class_idx][class_jdx].append(shared_ij_layer_k)

              # Get the candidates in layer for particular class
         # print("In layer {}, num shared globally: {}".format(idx, len(layer)))
         # print("Global shared out of layer average {} / {}".format(len(layer), layer_avg))
    print("Ranking by least coupled") 
    for i in range(num_groups):
        least_ranks = rank_by_least_coupled(shared_matrix, i)

    print("Ranked by most coupled")
    for i in range(num_groups):
        most_ranks = rank_by_most_coupled(shared_matrix, i)

def num_coupled(shared_matrix, i, j):
    total_coupled = sum(len(l) for l in shared_matrix[i][j])
    return total_coupled
    # print("Total channels coupled with class {} and class{} is : {}".format(i, j, total_coupled))

def rank_by_least_coupled(shared_matrix, i):
    scores = [(k, num_coupled(shared_matrix, i, k)) for k in range(num_groups) if k != i]
    sorted_scores = sorted(scores, key = lambda x:x[1])
    print(i, ": ", sorted_scores)
    return sorted_scores

def rank_by_most_coupled(shared_matrix, i):
    scores = [(k, num_coupled(shared_matrix, i, k)) for k in range(num_groups) if k != i]
    sorted_scores = sorted(scores, key = lambda x:x[1], reverse = True)
    print(i, ": ", sorted_scores)
    return sorted_scores

main()
