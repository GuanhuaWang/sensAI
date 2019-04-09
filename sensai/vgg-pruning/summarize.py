import argparse
import numpy as np

if __name__ == '__main__':
    # Load the lc data
    classes_layers_channels = []
    class_information = []
    for i in range(10):

        classes_layers_channels.append(np.load(open("lc_logs/class_{}_lc.npy".format(i), "rb")))
    
        summary_across_layer = []
        for index, layer in enumerate(classes_layers_channels[i]):
            layer_summary = {} 
            # Compute the absolute zeros for this layer
            non_zeros = [(i, layer[i]) for i in range(len(layer)) if layer[i] != 0]
            
            # Get boundary of 3 bins 
            non_zero_scores = [c[1] for c in non_zeros]
            percentile_bins = np.percentile(non_zero_scores, [3, 97])
            # Each non zero element now assigned to a bin
            indices = np.digitize(non_zero_scores, percentile_bins)

            # Record the channel index along with score
            bottom = [non_zeros[i] for i in range(len(indices)) if indices[i] == 0]
            top = [non_zeros[i] for i in range(len(indices)) if indices[i] == 2]

            bottom_indices = [non_zeros[i][0] for i in range(len(indices)) if indices[i] == 0]
            top_indices = [non_zeros[i][0] for i in range(len(indices)) if indices[i] == 2]


            layer_summary["Bottom"] = bottom
            layer_summary["Top"] = top

            layer_summary["Bottom_idx"] = bottom_indices
            layer_summary["Top_idx"] = top_indices
            summary_across_layer.append(layer_summary)

        class_information.append(summary_across_layer)

    # Print summary across layers for each class
    for j in range(16):
        print("Layer {}".format(j))
        for i in range(10):
            print("Class {}".format(i))
            print("Top X: ", class_information[i][j]["Top"])
            print("Bottom X: ", class_information[i][j]["Bottom"])


        print("")
        # This demonstrates the activation scores are different across classes, even though the filter activations may be shared
        print("Completely shared activation values (among all classes)")
        top_total = len(class_information[i][j]["Top"])
        bottom_total = len(class_information[i][j]["Bottom"])
        top_intersection = set.intersection(*[set(c_i) for c_i in [class_information[i][j]["Top"] for i in range(10)]])
        bottom_intersection = set.intersection(*[set(c_i) for c_i in [class_information[i][j]["Bottom"] for i in range(10)]])
        print("Ratio of top intersection to top total: {}/{}".format(len(top_intersection), top_total))
        print("Ratio of bottom intersection to bottom total: {}/{}".format(len(bottom_intersection), bottom_total))


        print("Completely shared filter indices (among all classes)")
        top_total = len(class_information[i][j]["Top"])
        bottom_total = len(class_information[i][j]["Bottom"])
        top_intersection = set.intersection(*[set(c_i) for c_i in [class_information[i][j]["Top_idx"] for i in range(10)]])
        bottom_intersection = set.intersection(*[set(c_i) for c_i in [class_information[i][j]["Bottom_idx"] for i in range(10)]])
        print("Ratio of top intersection to top total: {}/{}".format(len(top_intersection), top_total))
        print("Ratio of bottom intersection to bottom total: {}/{}".format(len(bottom_intersection), bottom_total))


        print("")
        print("")

