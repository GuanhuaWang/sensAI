import argparse
import numpy as np

"""
	Calculate the pruning candidates for a particular class i
"""
def pruning_candidates(class_index):
	layers_channels = np.load(open("lc_logs/class_{}_lc.npy".format(class_index), "rb"))
	candidates_by_layer = []
	print("Calculating pruning candidates for class {}".format(class_index))
	for index, layer in enumerate(layers_channels):
		candidates = [i for i in range(len(layer)) if layer[i] == 0]
		candidates_by_layer.append(candidates)
		print("Zero channels out of total in layer {}: {}/{}".format(index, len(candidates) ,len(layer)))
	return candidates_by_layer

"""
	Computes the hard zero pruning candidates across all 10 classes
	Pruning candidates for class i are saved in a prune_candidate_logs/class_i.npy file
	Data format list of list:
		data = np.load(open("prune_candidate_logs/class_{}.npy".format(i), "wb"))

	data := [[layer_1_channel_1_score, ..., layer_1_channel_k_score],
			[layer_2_channel_1_score, ..., layer_2_channel_l_score],
			...
			[layer_16_channel_1_Score, ..., layer_16_channel_y_score]]
"""
if __name__ == '__main__':
	candidates_across_classes = []
	for i in range(10):
		class_i_candidates = pruning_candidates(i)
		np.save(open("prune_candidate_logs/class_{}.npy".format(i), "wb"), class_i_candidates)
		candidates_across_classes.append(class_i_candidates)

	# Print channel indices with zero across classes and across layers
	for j in range(16):
		print("Layer {}".format(j))
		for i in range(10):
			print("Class {}".format(i))
			print(candidates_across_classes[i][j])