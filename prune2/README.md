
1) Specify pruning parameters in apoz_policy.py, activations.py <br>
   Specify size of activation set in dataset.py <br>
   [Can make program arguments that take parameters if heavy parameter tuning is needed] <br>

2) Specify group indices in ./scripts/activations_grouped_vgg19.sh <br>

3) To generate pruning candidates run <br>
   ./scripts/activations_grouped_vgg19.sh  <br>

4) Pruning candidate now stored in ./prune_candidate_logs <br>

5) Generate pruned model, run   <br>
   python3 prune_and_get_model.py -a vgg19_bn --resume ./checkpoint_bearclaw.pth.tar  -c ./prune_candidate_logs/ -s ./TO_SAVE_MODEL_BASE_DIR <br>
   Models now saved at location, <br>
      ./TO_SAVE_MODEL_BASE_DIR <br>

6) Specify from dir (pruned models) and save dir (pruned and retrained models), and training parameters in, <br>
   ./scripts/train_pruned_grouped.sh <br>

After above script runs, retrained model located at specified save dir <br>

7) To evaluate, <br>

  python3 evaluate.py -a vgg19_bn --test-batch 100  --resume ./PATH_TO_RETRAIN_SAVED_DIR/ --evaluate --grouped

