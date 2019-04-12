import argparse
import numpy as np
import torch
import io


"""
	Calculate the Average Percentage of Zeros Score of the feature map activation layer output
"""
def apoz_scoring(activation):
    activation = activation.cpu()
    if activation.dim() == 4:
        view_2d = activation.view(-1, activation.size(2) * activation.size(3))  # (batch*channels) x (h*w)
        featuremap_apoz = view_2d.abs().gt(0.01).sum(dim=1).float() / (activation.size(2) * activation.size(3))  # (batch*channels) x 1
        featuremap_apoz_mat = featuremap_apoz.view(activation.size(0), activation.size(1))  # batch x channels
    elif activation.dim() == 2:
        featuremap_apoz_mat = activation.abs().gt(0.01).sum(dim=1).float() / activation.size(1)  # batch x 1
    else:
        raise ValueError("activation_channels_apoz: Unsupported shape: ".format(activation.shape))
    return 100 - featuremap_apoz_mat.mean(dim=0).mul(100).cpu()

def apoz_avg(activation):
	activation = activation.cpu()
	if activation.dim() == 4:
		view_2d = activation.view(-1, activation.size(2) * activation.size(3))
		featuremap_avg = view_2d.abs().sum(dim = 1).float() / (activation.size(2) * activation.size(3))
		featuremap_avg_mat = featuremap_avg.view(activation.size(0), activation.size(1))
	elif activation.dim() == 2:
		featuremap_avg_mat = activation.abs().sum(dim = 1).float() / activation.size(1)
	else:
		raise ValueError("activation_channels_avg: Unsupported shape: ".format(activation.shape))
	return featuremap_avg_mat.mean(dim = 0).cpu()

def pruning_candidates(class_index, thresholds, avg_thresholds):
	layers_channels = []
	fmap_file =  open("/home/ubuntu/baseModel/feature_map_data/class_{}_fmap.pt".format(class_index), "rb")
	data_buffer = io.BytesIO(fmap_file.read())
	for _ in range(16):
		layers_channels.append(torch.load(data_buffer))

	candidates_by_layer = []

	total_channels = sum([layer.shape[1] for layer in layers_channels])
	print(total_channels)
	num_candidates = 0
	print("Calculating pruning candidates for class {}".format(class_index))
	for index, layer in enumerate(layers_channels):
		apoz_score = apoz_scoring(layer)
		avg_score = apoz_avg(layer)
		# prune_intersection = list(set(apoz_score).intersection(set(avg_score)))
		# if index == 0:
		#	print(apoz_score)
		#	print(avg_score)
		#	print(prune_intersection)

		print(apoz_score.mean())

		curr_threshold = thresholds[index]
		while True:
			num_candidates = apoz_score.gt(curr_threshold).sum()
			print("Greater than {} %".format(curr_threshold), num_candidates)
			if num_candidates < apoz_score.size()[0]:
				candidates = [x[0] for x in apoz_score.gt(curr_threshold).nonzero().tolist()]
				break
			curr_threshold += 5
		

		avg_candidates = [idx for idx, score in enumerate(avg_score) if score >= avg_thresholds[index]]	
		print(avg_candidates)
		print(candidates)
		difference_candidates = list(set(candidates).difference(set(avg_candidates)))
		print(difference_candidates)
		num_candidates += len(difference_candidates)		

		print("Class Index: {}, Layer {}, Number of neurons with hybrid > {}%: {}/{}".format(class_index, index, curr_threshold, len(difference_candidates), apoz_score.size()[0]))
		candidates_by_layer.append(difference_candidates)
	print("number of pruning channels out of total: {}/{}".format(num_candidates,total_channels))
	return candidates_by_layer, float(num_candidates) /total_channels



if __name__ == '__main__':
	candidates_across_classes = []
	## Specificy threshold by layer
	# apoz_thresh_candidates = [90] #[10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
	# avg_thresh_candidates = [0.001] #[j*10**i for i in range(-4, 2) for j in [ 1, 2, 5]] 
	apoz_thresh = 80
	avg_thresh = .01
	thresh_log_file = open("thresh_logs.txt", "w")

	for i in range(10):
		apoz_thresholds = [apoz_thresh] * 16
		avg_thresholds = [avg_thresh] * 16
		class_i_candidates, percent_neurons = pruning_candidates(i, apoz_thresholds, avg_thresholds)
		print("Percentage of neurons pruned with apoz threshold {}, avg threshold {}, percentage {}".format(apoz_thresh, avg_thresh, percent_neurons))
		np.save(open("prune_candidate_logs/class_{}_apoz_layer_thresholds.npy".format(i), "wb"), class_i_candidates)
	""" 
	for apoz_thresh in apoz_thresh_candidates:
		for avg_thresh in avg_thresh_candidates:
			thresholds = [apoz_thresh] * 16
			avg_thresholds = [avg_thresh] * 16
			prune_candidates, percentage_pruned = pruning_candidates(0, thresholds, avg_thresholds) 
			if percentage_pruned >= .50:
				thresh_log_file.write(str(apoz_thresh) + " " + str(avg_thresh))	
				np.save(open("prune_candidate_logs/search/class_0_apoz{}_avg{}.npy".format(apoz_thresh, avg_thresh), "wb"), prune_candidates)

	thresh_log_file.close()
	"""

	"""
	for i in range(10):
		class_i_candidates, percent_neurons = pruning_candidates(i, thresholds, avg_thresholds)
		print("Percentage of neurons pruned with apoz threshold {}, avg threshold {}, percentage {}".format(apoz_thresh, avg_thresh, percent_neurons))
		# Verify the types
		# print(type(class_i_candidates), type(class_i_candidates[0]), type(class_i_candidates[0][0]))
		np.save(open("prune_candidate_logs/class_{}_apoz_layer_thresholds.npy".format(i), "wb"), class_i_candidates)
		class_i_candidates.append(class_i_candidates)
		# Test data was stored correctly,
		# data = np.load(open("prune_candidate_logs/class_{}.npy".format(i), "rb"))
		# for idx, layer in enumerate(data):
		#	print(data)
	"""
	# summarize_candidates(candidates_across_classes))

