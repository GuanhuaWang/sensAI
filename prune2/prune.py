import torch
from torchvision import models
import numpy as np
 
def replace_layers(model, i, indexes, layers):
    if i in indexes:
        return layers[indexes.index(i)]
    return model[i]

def prune_vgg16_conv_layer(model, layer_index, filter_index, use_batch_norm=False):
    _, conv = list(model.features._modules.items())[layer_index]
    next_conv = None
    offset = 1

    while layer_index + offset <  len(model.features._modules.items()):
        res =  list(model.features._modules.items())[layer_index+offset]
        if isinstance(res[1], torch.nn.modules.conv.Conv2d):
            next_name, next_conv = res
            break
        offset = offset + 1
    
    new_conv = \
        torch.nn.Conv2d(in_channels = conv.in_channels, \
            out_channels = conv.out_channels - 1,
            kernel_size = conv.kernel_size, \
            stride = conv.stride,
            padding = conv.padding,
            dilation = conv.dilation,
            groups = conv.groups,
            bias = True)#conv.bias)

    old_weights = conv.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()
    new_weights[:filter_index, :, :, :] = old_weights[:filter_index, :, :, :]
    new_weights[filter_index : , :, :, :] = old_weights[filter_index + 1 :, :, :, :]
    new_conv.weight.data = torch.from_numpy(new_weights).cuda()

    if conv.bias is not None:
        bias_numpy = conv.bias.data.cpu().numpy()
        bias = np.zeros(shape = (bias_numpy.shape[0] - 1), dtype = np.float32)
        bias[:filter_index] = bias_numpy[:filter_index]
        bias[filter_index : ] = bias_numpy[filter_index + 1 :]
        new_conv.bias.data = torch.from_numpy(bias).cuda()

    if use_batch_norm:
        _, bn = list(model.features._modules.items())[layer_index + 1]
        new_bn = torch.nn.BatchNorm2d(conv.out_channels - 1)

        old_weights = bn.weight.data.cpu().numpy()
        new_weights = new_bn.weight.data.cpu().numpy()
        new_weights[:filter_index] = old_weights[:filter_index]
        new_weights[filter_index:] = old_weights[filter_index+1:]
        

        old_bias = bn.bias.data.cpu().numpy()
        new_bias = new_bn.bias.data.cpu().numpy()
        new_bias[:filter_index] = old_bias[:filter_index]
        new_bias[filter_index:] = old_bias[filter_index+1:]

        

        old_running_mean = bn.running_mean.data.cpu().numpy()
        new_running_mean = new_bn.running_mean.data.cpu().numpy()
        new_running_mean[:filter_index] = old_running_mean[:filter_index]
        new_running_mean[filter_index:] = old_running_mean[filter_index+1:]


        old_running_var = bn.running_var.data.cpu().numpy()
        new_running_var = new_bn.running_var.data.cpu().numpy()
        new_running_var[:filter_index] = old_running_var[:filter_index]
        new_running_var[filter_index:] = old_running_var[filter_index+1:]

        new_bn.weight.data = torch.from_numpy(new_weights).cuda()
        new_bn.bias.data = torch.from_numpy(new_bias).cuda()
        new_bn.running_mean.data = torch.from_numpy(new_running_mean).cuda()
        new_bn.running_var.data = torch.from_numpy(new_running_var).cuda()
        

    if not next_conv is None:
        next_new_conv = \
            torch.nn.Conv2d(in_channels = next_conv.in_channels - 1,\
                out_channels =  next_conv.out_channels, \
                kernel_size = next_conv.kernel_size, \
                stride = next_conv.stride,
                padding = next_conv.padding,
                dilation = next_conv.dilation,
                groups = next_conv.groups,
                bias = True)#next_conv.bias)

        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = next_new_conv.weight.data.cpu().numpy()

        new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
        new_weights[:, filter_index : , :, :] = old_weights[:, filter_index + 1 :, :, :]
        next_new_conv.weight.data = torch.from_numpy(new_weights).cuda()

        if next_conv.bias is not None:
            next_new_conv.bias.data = torch.from_numpy(next_conv.bias.data.cpu().numpy().copy()).cuda() 

    if not next_conv is None:
        features = torch.nn.Sequential(
                *(replace_layers(model.features, i, [layer_index, layer_index + 1, layer_index+offset], \
                    [new_conv, new_bn, next_new_conv]) for i, _ in enumerate(model.features)))
        del model.features
        del conv

        model.features = features
    else:
        #Prunning the last conv layer. This affects the first linear layer of the classifier.
        model.features = torch.nn.Sequential(
                *(replace_layers(model.features, i, [layer_index, layer_index+1], \
                    [new_conv, new_bn]) for i, _ in enumerate(model.features)))
        layer_index = 0
        old_linear_layer = None
        if len(model.classifier._modules):
            for _, module in model.classifier._modules.items():
                if isinstance(module, torch.nn.Linear):
                    old_linear_layer = module
                    break
                layer_index = layer_index  + 1
        else:
            old_linear_layer = model.classifier

        if old_linear_layer is None:
            raise BaseException("No linear layer found in classifier")
        params_per_input_channel = old_linear_layer.in_features / conv.out_channels

        new_linear_layer = \
            torch.nn.Linear(int(old_linear_layer.in_features - params_per_input_channel), 
                int(old_linear_layer.out_features))
        
        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()        

        new_weights[:, : int(filter_index * params_per_input_channel)] = \
            old_weights[:, : int(filter_index * params_per_input_channel)]
        new_weights[:, int(filter_index * params_per_input_channel) :] = \
            old_weights[:, int((filter_index + 1) * params_per_input_channel) :]
        
        new_linear_layer.bias.data = torch.from_numpy(old_linear_layer.bias.data.cpu().numpy()).cuda()

        new_linear_layer.weight.data = torch.from_numpy(new_weights).cuda()

        if len(model.classifier._modules):
            classifier = torch.nn.Sequential(
                *(replace_layers(model.classifier, i, [layer_index], \
                    [new_linear_layer]) for i, _ in enumerate(model.classifier)))
        else:
            classifier = torch.nn.Sequential(new_linear_layer)

        del model.classifier
        del next_conv
        del conv
        model.classifier = classifier

    return model

def prune_last_fc_layer(model, class_indices, use_bce=False):
    layer_index = 0
    old_linear_layer = None
    counter = 0
    if len(model.classifier._modules):
        for _, module in model.classifier._modules.items():
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                layer_index = counter
            counter += 1
    else:
        old_linear_layer = model.classifier

    if old_linear_layer is None:
            raise BaseException("No linear layer found in classifier")

    bce_offset = 0 if use_bce else 1
    new_linear_layer = \
        torch.nn.Linear(int(old_linear_layer.in_features), 
            len(class_indices) + bce_offset)
    
    old_weights = old_linear_layer.weight.data.cpu().numpy()
    new_weights = new_linear_layer.weight.data.cpu().numpy()        

    new_weights[bce_offset:, :] = old_weights[class_indices, :]
    
    new_linear_layer.bias.data[bce_offset:] = torch.from_numpy(np.asarray(old_linear_layer.bias.data.cpu().numpy()[class_indices])).cuda()

    new_linear_layer.weight.data = torch.from_numpy(new_weights).cuda()

    if len(model.classifier._modules):
        classifier = torch.nn.Sequential(
            *(replace_layers(model.classifier, i, [layer_index], \
                [new_linear_layer]) for i, _ in enumerate(model.classifier)))
    else:
        classifier = torch.nn.Sequential(new_linear_layer)

    del model.classifier
    model.classifier = classifier

    return model

if __name__ == '__main__':
    model = models.vgg16(pretrained=True)
    model.train()

    t0 = time.time()
    model = prune_conv_layer(model, 28, 10)
    print("The prunning took", time.time() - t0)
