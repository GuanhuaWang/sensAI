#!/usr/bin/env python3

import argparse
import json
import os
import yaml

import torch

import group_selection
import get_prune_candidates
import prune_and_get_model
import evaluate

parser = argparse.ArgumentParser(description='sensAI auto script')

parser.add_argument('config_file', type=str, help="the path of the yaml config file")
parser.add_argument('--force-override', action='store_true',
                    help="rerun the whole program and override existing results")
parser.add_argument('--skip-retrain', action='store_true',
                    help="skip retraining the models and only evaluate pruned models")
args = parser.parse_args()
with open(args.config_file) as f:
    config = yaml.load(f)

seed = config['seed']
dataset = config['dataset']
arch = config['arch']
pretrained_model = config['pretrained_model']
use_cuda = torch.cuda.is_available() and True
n_dataset_loading_workers = config['n_dataset_loading_workers']

# seed our program
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed_all(seed)
# torch.set_printoptions(threshold=10000)

project_dir = os.path.join("outputs", config["project"])
os.makedirs(project_dir, exist_ok=True)

# grouping
grouping_result_file = os.path.join(project_dir, "grouping_results.json")
if not os.path.exists(grouping_result_file) or args.force_override:
    print("=> start grouping classes")
    grouping_result = group_selection.group_classes(
        config['n_groups'], dataset, arch, pretrained_model, seed, use_cuda,
        group_selection_algorithm=config['group_selection_algorithm'],
        n_workers=n_dataset_loading_workers)
    with open(grouping_result_file, 'w') as f:
        print(f"=> saving grouping results to '{grouping_result_file}'")
        json.dump(grouping_result, f, indent=4)
else:
    print(f"=> grouping results exist ({grouping_result_file}). grouping skipped")
    with open(grouping_result_file) as f:
        grouping_result = json.load(f)

# generate pruning candidates
pruning_candidates_file = os.path.join(project_dir, "pruning_candidates.json")
if not os.path.exists(pruning_candidates_file) or args.force_override:
    print("=> start generating pruning candidates")
    candidates_from_pretrained = get_prune_candidates.get_candidates_from_pretrained(
        dataset, arch, pretrained_model,
        grouping_result, use_cuda,
        n_workers=n_dataset_loading_workers)
    with open(pruning_candidates_file, 'w') as f:
        print(f"=> saving pruning candidates to '{pruning_candidates_file}'")
        json.dump(candidates_from_pretrained, f, indent=4)
else:
    print(f"=> pruning candidates exist ({pruning_candidates_file}). pruning candidates generation skipped")
    with open(pruning_candidates_file) as f:
        candidates_from_pretrained = json.load(f)

# prune and get model
pruned_models_dir = os.path.join(project_dir, 'pruned_models')
if not os.path.exists(pruned_models_dir) or args.force_override:
    print("=> start generating pruned models")
    os.makedirs(pruned_models_dir)
    prune_and_get_model.prune_model_from_pretrained(dataset, arch, pretrained_model, grouping_result,
                                                    candidates_from_pretrained, pruned_models_dir, use_cuda)
else:
    print(f"=> pruned models exist ({pruned_models_dir}). pruning candidates generation skipped")

# retrain
retrained_models_dir = os.path.join(project_dir, 'retrained_models')
if not args.skip_retrain:
    if not os.path.exists(retrained_models_dir) or args.force_override:
        raise NotImplementedError
        print("=> start retraining models")
        os.makedirs(retrained_models_dir)
    else:
        print(f"=> retrained models exist ({retrained_models_dir}). retrained candidates generation skipped")
else:
    print(f"=> retraining is skipped by the user")

# evaluate
if args.skip_retrain:
    models_dir = pruned_models_dir
else:
    models_dir = retrained_models_dir
evaluate.evaluate_models(dataset, models_dir, grouping_result, use_cuda, n_workers=n_dataset_loading_workers)
