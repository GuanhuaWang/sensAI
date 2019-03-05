import argparse
import numpy as np

def pruning_candidates(class_index):
	layers_channels = np.load(open("lc_logs/class_{}_lc.npy".format(class_index), "rb"))
	candidates_by_layer = []
	print("Calculating pruning candidates for class {}".format(class_index))
	for index, layer in enumerate(layers_channels):
		candidates = [i for i in range(len(layer)) if layer[i] == 0]
		candidates_by_layer.append(candidates)
		print("Zero channels out of total in layer {}: {}/{}".format(index, len(candidates) ,len(layer)))
	return candidates_by_layer

if __name__ == '__main__':
	for i in range(10):
		class_i_candidates = pruning_candidates(i)
		# Verify the types
		# print(type(class_i_candidates), type(class_i_candidates[0]), type(class_i_candidates[0][0]))
		np.save(open("prune_candidate_logs/class_{}.npy".format(i), "wb"), class_i_candidates)
		
		# Test data was stored correctly,
		# data = np.load(open("prune_candidate_logs/class_{}.npy".format(i), "rb"))
		# for idx, layer in enumerate(data):
		# 	print(data)
