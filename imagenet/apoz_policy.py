import argparse
import numpy as np
import torch
import io
import glob 
import re

path = './feature_map_data/'
file_names = [f for f in glob.glob(path + "*.pt", recursive=False)]
group_id_list = [re.search('\(([^)]+)', f_name).group(1) for f_name in file_names]

"""
	Calculate the Average Percentage of Zeros Score of the feature map activation layer output
"""
def apoz_scoring(activation):
    # activation = activation.cpu()
    if activation.dim() == 4:
        view_2d = activation.view(-1, activation.size(2) * activation.size(3))  # (batch*channels) x (h*w)
        featuremap_apoz = view_2d.abs().gt(0.005).sum(dim=1).float() / (activation.size(2) * activation.size(3))  # (batch*channels) x 1
        featuremap_apoz_mat = featuremap_apoz.view(activation.size(0), activation.size(1))  # batch x channels
    elif activation.dim() == 2:
        featuremap_apoz_mat = activation.abs().gt(0.005).sum(dim=1).float() / activation.size(1)  # batch x 1
    else:
        raise ValueError("activation_channels_apoz: Unsupported shape: ".format(activation.shape))
    return 100 - featuremap_apoz_mat.mean(dim=0).mul(100)


def avg_scoring(activation):
	# activation = activation.cpu()
	if activation.dim() == 4:
		view_2d = activation.view(-1, activation.size(2) * activation.size(3))
		featuremap_avg = view_2d.abs().sum(dim = 1).float() / (activation.size(2) * activation.size(3))
		featuremap_avg_mat = featuremap_avg.view(activation.size(0), activation.size(1))
	elif activation.dim() == 2:
		featuremap_avg_mat = activation.abs().sum(dim = 1).float() / activation.size(1)
	else:
		raise ValueError("activation_channels_avg: Unsupported shape: ".format(activation.shape))
	return featuremap_avg_mat.mean(dim = 0)

def pruning_candidates(group_id, thresholds, file_name):
	layers_channels = []
	fmap_file = open(file_name, "rb")
	data_buffer = io.BytesIO(fmap_file.read())
	for _ in range(16):
		layers_channels.append(torch.load(data_buffer))

	candidates_by_layer = []
	print("Calculating pruning candidates for classe(s) {}".format(group_id))
	for index, layer in enumerate(layers_channels):
		apoz_score = apoz_scoring(layer)
		print(apoz_score.mean())

		curr_threshold = thresholds[index]
		while True:
			num_candidates = apoz_score.gt(curr_threshold).sum()
			print("Greater than {} %".format(curr_threshold), num_candidates)
			if num_candidates < apoz_score.size()[0]:
				candidates = [x[0] for x in apoz_score.gt(curr_threshold).nonzero().tolist()]
				break
			curr_threshold += 5

		print("Class Index: {}, Layer {}, Number of neurons with apoz > {}%: {}/{}".format(group_id, index, curr_threshold, len(candidates), apoz_score.size()[0]))
		candidates_by_layer.append(candidates)
		print("Zero channels out of total in layer {}: {}/{}".format(index, len(candidates) ,len(layer)))
	return candidates_by_layer


if __name__ == '__main__':
	candidates_across_classes = []
	## Specificy threshold by layer
	thresholds = [55] * 16
	for group_id, file_name in zip(group_id_list, file_names):
		group_candidates = pruning_candidates(group_id, thresholds, file_name)
		np.save(open("prune_candidate_logs/class_({})_apoz_layer_thresholds.npy".format(group_id), "wb"), group_candidates)
		candidates_across_classes.append(group_candidates)
	
