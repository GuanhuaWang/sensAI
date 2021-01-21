<p align="center">
  <img src="sensAI-logo.png"  width="250" height="250">
</p>

# sensAI: ConvNets Decomposition via Class Parallelism for Fast Inference on Live Data

## Environment

Linux, python 3.6+

## Setup

```bash
pip install -r requirements.txt
```

## Instruction

Supported CNN architectures and datasets:

| Dataset        | Architecture(`ARCH`) |
| -------------  |:-------------:|
| CIFAR-10       | vgg19_bn, resnet110, resnet164, mobilenetv2, shufflenetv2|
| CIFAR-100      | vgg19_bn, resnet110, resnet164|
| ImageNet-1K    | vgg19_bn, resnet50|


### 1. Generate class groups
   
   For CIFAR-10/CIFAR-100:
   ```bash
   python3 group_selection.py \
           --arch $ARCH \
           --resume $pretrained_model \
           --dataset $DATASET \
           --ngroups $number_of_groups \
           --gpu_num $number_of_gpu 
   ```
   For ImageNet-1K:
   ```bash
   python3 group_selection.py \
           --arch $ARCH \
           --dataset imagenet \
           --ngroups $number_of_groups \
           --gpu_num $number_of_gpu \
           --data /{path_to_imagenet_dataset}/
   ```
   
   Pruning candidate now stored in `./prune_candidate_logs/`
   
### 2. Prune models
    
   For CIFAR-10/CIFAR-100:
   ```bash
   python3 prune_and_get_model.py \
           -a $ARCH \
           --dataset $DATASET \
           --resume $pretrained_model \
           -c ./prune_candidate_logs/ \
           -s ./{TO_SAVE_PRUNED_MODEL_DIR}/
   ```
   For ImageNet-1K:
   ```bash
   python3 prune_and_get_model.py \
           -a $ARCH \
           --dataset imagenet \
           -c ./prune_candidate_logs/ \
           -s ./{TO_SAVE_PRUNED_MODEL_DIR}/ \
           --pretrained
   ```
   
   Pruned models are now saved in `./{TO_SAVE_PRUNED_MODEL_DIR}/$ARCH/`
   
### 3. Retrain pruned models
  
   For CIFAR-10/CIFAR-100:
   ```bash
   python3 retrain_grouped_model.py \
           -a $ARCH \
           --dataset $DATASET \
           --resume ./{TO_SAVE_PRUNED_MODEL_DIR}/ \
           --train_batch $batch_size \
           --epochs $number_of_epochs \
           --num_gpus $number_of_gpus
   ```
   For ImageNet-1K:
   ```bash
   python3 retrain_grouped_model.py \
           -a $ARCH \
           --dataset imagenet \
           --resume ./{TO_SAVE_PRUNED_MODEL_DIR}/ \
           --epochs $number_of_epochs \
           --num_gpus $number_of_gpus \
           --train_batch $batch_size \
           --data /{path_to_imagenet_dataset}/
   ```
   
   Retrained models now saved in `./{TO_SAVE_PRUNED_MODEL_DIR}_retrained/$ARCH/`
   
### 4. Evaluate

   For CIFAR-10/CIFAR-100:
   ```bash
   python3 evaluate.py \
           -a $ARCH \
           --dataset=$DATASET \
           --retrained_dir ./{TO_SAVE_PRUNED_MODEL_DIR}_retrained/ \
           --test-batch $batch_size
   ```
   For ImageNet-1K:
   ```bash
   python3 evaluate.py \
           -d imagenet \
           -a $ARCH \
           --retrained_dir ./{TO_SAVE_PRUNED_MODEL_DIR}_retrained/ \
           --data /{path_to_imagenet_dataset}/
   ```

## Contributors

Thanks for all the main contributors to this repository:

* [Brandon Hsieh](https://github.com/hsiehbrandon) 

* [Zhuang Liu](https://github.com/liuzhuang13)

* [Kenan Jiang](https://github.com/Kenan-Jiang) 

* [Kehan Wang](https://github.com/Jason-Khan)

* [Siyuan Zhuang](https://github.com/suquark)

And many others [Zihao Fan](https://github.com/zihao-fan), [Hank O'Brien](https://github.com/hjobrien) , [Yaoqing Yang](https://github.com/nsfzyzz), [Adarsh Karnati](https://github.com/akarnati11), [Jichan Chung](https://github.com/jichan3751), [Yingxin Kang](https://github.com/Miiira), [
Balaji Veeramani](https://github.com/bveeramani), [Sahil Rao](https://github.com/sahilrao21).




## Citation

```text
@inproceedings{wang2021sensAI,
 author = {Guanhua Wang and Zhuang Liu and Brandon Hsieh and Siyuan Zhuang and Joseph Gonzalez and Trevor Darrell and Ion Stoica},
 title = {{sensAI: ConvNets Decomposition via Class Parallelism for Fast Inference on Live Data}},
 booktitle = {Proceedings of Fourth Conference on Machine Learning and Systems (MLSys'21)},
 year = {2021}
} 
```

