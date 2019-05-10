
1) Specify pruning parameters in apoz_policy.py, activations.py
   Specify size of activation set in dataset.py
   [Can make program arguments that take parameters if heavy parameter tuning is needed]

2) Specify group indices in ./scripts/activations_grouped_vgg19.sh

3) To generate pruning candidates run
   ./scripts/activations_grouped_vgg19.sh

4) Pruning candidate now stored in ./prune_candidate_logs

5) Generate pruned model, run
   python3 prune_and_get_model.py -a vgg19_bn --resume ./checkpoint_bearclaw.pth.tar  -c ./prune_candidate_logs/ -s ./TO_SAVE_MODEL_BASE_DIR
   Models now saved at location,
      ./TO_SAVE_MODEL_BASE_DIR

6) Specify from dir (pruned models) and save dir (pruned and retrained models), and training parameters in,
   ./scripts/train_pruned_grouped.sh

After above script runs, retrained model located at specified save dir

7) To evaluate,

  python3 evaluate.py -a vgg19_bn --test-batch 100  --resume ./PATH_TO_RETRAIN_SAVED_DIR/ --evaluate --grouped

