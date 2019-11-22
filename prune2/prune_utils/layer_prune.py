def prune_linear_layer(linear_layer, class_indices):
    class_indices = class_indices.copy()
    class_indices += [0]  # use 0 as the extra class
    linear_layer.bias.data = linear_layer.bias.data[class_indices]
    linear_layer.weight.data = linear_layer.weight.data[class_indices, :]
    linear_layer.out_features = len(class_indices)
    return linear_layer