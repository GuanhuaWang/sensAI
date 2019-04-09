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
        featuremap_apoz = view_2d.abs().sum(dim = 1) / (activation.size(2) * activation.size(3))
        featuremap_apoz_mat = featuremap_apoz.view(activation.size(0), activation.size(1))  # batch x channels
    elif activation.dim() == 2:
        featuremap_apoz_mat = activation.abs().sum(dim = 1) / activation.size(1)
    else:
        raise ValueError("activation_channels_apoz: Unsupported shape: ".format(activation.shape))
    return featuremap_apoz_mat.mean(dim=0).cpu()

def pruning_candidates(class_index, thresholds):
    layers_channels = []
    fmap_file =  open("/home/ubuntu/baseModel/feature_map_data/class_{}_fmap.pt".format(class_index), "rb")
    data_buffer = io.BytesIO(fmap_file.read())
    for _ in range(16):
        layers_channels.append(torch.load(data_buffer))

    candidates_by_layer = []
    print("Calculating pruning candidates for class {}".format(class_index))
    for index, layer in enumerate(layers_channels):
        apoz_score = apoz_scoring(layer)
        print("layer", index, "mean:", apoz_score.mean())
        print("layer", index, "std:", apoz_score.std())
        curr_threshold = thresholds[index]
        while True:
            num_candidates = apoz_score.lt(curr_threshold).sum()
            print("Less than {} :".format(curr_threshold), num_candidates)
            if num_candidates < apoz_score.size()[0]:
                candidates = [x[0] for x in apoz_score.lt(curr_threshold).nonzero().tolist()]
                break
            curr_threshold -= 0.005

        print("Class Index: {}, Layer {}, Number of neurons with avg weight < {}: {}/{}".format(class_index, index, curr_threshold, len(candidates), apoz_score.size()[0]))
        candidates_by_layer.append(candidates)
    return candidates_by_layer

if __name__ == '__main__':
    candidates_across_classes = []
    ## Specificy threshold by layer
    thresholds = [0.005] * 16
    for i in range(10):
        class_i_candidates = pruning_candidates(i, thresholds)
        # Verify the types
        # print(type(class_i_candidates), type(class_i_candidates[0]), type(class_i_candidates[0][0]))
        np.save(open("prune_candidate_logs/class_{}_apoz_layer_thresholds.npy".format(i), "wb"), class_i_candidates)
        class_i_candidates.append(class_i_candidates)
        # Test data was stored correctly,
        # data = np.load(open("prune_candidate_logs/class_{}.npy".format(i), "rb"))
        # for idx, layer in enumerate(data):
        #   print(data)
    # summarize_candidates(candidates_across_classes))

