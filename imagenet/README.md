1) To gather candidates first specify the arch and num gpus in the file, 
     `gather_candidates.py` 

    If want to use a different grouping scheme, specify in `gather_candidates.py` as well. 
    Once complete, candidates will be stored in `./prune_candidate_logs`

    Note: If debugging and want to generate candidates faster, can specify less sample points to generate for activations in imagenet_dataset.py, under the activations case in the init of the image folder. 

2) To generate pruned models specify the prune model save dir and arch with the following script,
    
    For VGG:
    `python3 prune_and_get_model.py -a vgg19_bn --pretrained -c ./prune_candidate_logs/ -s ./PRUNED_MODEL_SAVE_DIR`

    For ResNet:
    `python3 prune_and_get_resnet.py -a resnet50 --pretrained -c ./prune_candidate_logs/ -s ./PRUNED_MODEL_SAVE_DIR` 

    Once complete, pruned models will be stored in `./PRUNED_MODEL_SAVE_DIR` 
    The grouping config will also be saved in the same directory
    The grouping config is nesscary for associating which classes to which groups 

3) To retrain, specify the following in `./scripts/train_pruned_grouped.sh` 
    EPOCHS=  
    FROM=./PRUNED_MODEL_SAVE_DIR 
    ARCH=

    Then run `./scripts/train_pruned_grouped.sh` 

    The retrained models will be saved in `./PRUNED_MODEL_SAVE_DIR_retrained/`
    Note: The current setup trains each model sequentially 

4) For evaluation, to evaluate a single group 
     
     `python3 imagenet_official_retrain.py /home/ubuntu/imagenet --arch vgg19_bn --resume \
                 PATH_TO_SAVED_MODEL/vgg19_bn/MODEL_PATH.pth --evaluate \
                 --config pruned_models_test/grouping_config.npy`
                 
    To evaluate the recombined group classifiers, 
    `python3 imagenet_evaluate_grouped.py ~/imagenet/ -a vgg19_bn --resume ./pruned_models_30M_param_retrained/ --evaluate --config  ./pruned_models_30M_param/grouping_config.npy`
