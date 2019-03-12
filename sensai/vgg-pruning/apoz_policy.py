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
        featuremap_apoz = view_2d.abs().gt(0).sum(dim=1).float() / (activation.size(2) * activation.size(3))  # (batch*channels) x 1
        featuremap_apoz_mat = featuremap_apoz.view(activation.size(0), activation.size(1))  # batch x channels
    elif activation.dim() == 2:
        featuremap_apoz_mat = activation.abs().gt(0).sum(dim=1).float() / activation.size(1)  # batch x 1
    else:
        raise ValueError("activation_channels_apoz: Unsupported shape: ".format(activation.shape))
    return 100 - featuremap_apoz_mat.mean(dim=0).mul(100).cpu()

def pruning_candidates(class_index):
	layers_channels = []
	fmap_file =  open("feature_map_data/class_{}_fmap.pt".format(class_index), "rb")
	data_buffer = io.BytesIO(fmap_file.read())
	for _ in range(16):
		layers_channels.append(torch.load(data_buffer))

	candidates_by_layer = []
	print("Calculating pruning candidates for class {}".format(class_index))
	for index, layer in enumerate(layers_channels):
		apoz_score = apoz_scoring(layer)
		print(apoz_score.mean())
		print("Greater than 90%", apoz_score.gt(90).sum())
		print("Layer {}, Number of neurons with apoz > 90%: {}/{}".format(class_index, apoz_score.gt(90).sum(), apoz_score.size()[0]))
		candidates = [x[0] for x in apoz_score.gt(90).nonzero().tolist()]
		candidates_by_layer.append(candidates)
		# print("Zero channels out of total in layer {}: {}/{}".format(index, len(candidates) ,len(layer)))
	return candidates_by_layer

if __name__ == '__main__':
	candidates_across_classes = []
	for i in range(10):
		class_i_candidates = pruning_candidates(i)
		# Verify the types
		# print(type(class_i_candidates), type(class_i_candidates[0]), type(class_i_candidates[0][0]))
		np.save(open("prune_candidate_logs/class_{}_apoz.npy".format(i), "wb"), class_i_candidates)
		class_i_candidates.append(class_i_candidates)
		# Test data was stored correctly,
		# data = np.load(open("prune_candidate_logs/class_{}.npy".format(i), "rb"))
		# for idx, layer in enumerate(data):
		# 	print(data)
	# summarize_candidates(candidates_across_classes))
