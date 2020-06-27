import argparse
import os
import glob
import subprocess as sp
import numpy as np
parser = argparse.ArgumentParser(description='retrain pruned model')
parser.add_argument('-d', '--dataset', required=True, type=str)
parser.add_argument('--epochs', required=True, type=int)
parser.add_argument('-a', '--arch', default='vgg19_bn',
                    type=str, help='The architecture of the trained model')
parser.add_argument('-r', '--resume', default='', type=str,    
                    help='The path to the checkpoints')    ### pruned models are saved here
parser.add_argument('--num_gpus', default=4, type=int)
parser.add_argument('--train_batch', default=256, type=int)
parser.add_argument('--data', default='/home/ubuntu/imagenet', required=False, type=str,
                    help='location of the imagenet dataset that includes train/val')

args = parser.parse_args()


def main():
    save = args.resume[:-1] +'_retrained/'
    groups = np.load(open(args.resume + "grouping_config.npy", "rb"))
    resultExist = os.path.exists(save)
    if resultExist:
        rm_cmd = 'rm -rf ' + save
        sp.Popen(rm_cmd, shell=True)
    os.mkdir(save)
    np.save(open(os.path.join(save[:-1], "grouping_config.npy"), "wb"), groups)
    save += args.arch
    os.mkdir(save)
    files = [f for f in glob.glob(args.resume + args.arch+"/*.pth", recursive=False)]
    process_list = [None for _ in range(args.num_gpus)]
    if args.dataset in ['cifar10', 'cifar100']:
        for i, file in enumerate(files):
            if process_list[i % args.num_gpus]:
                process_list[i % args.num_gpus].wait()
            exec_cmd = 'python3 cifar_group.py' +\
                            ' --arch %s' % args.arch +\
                            ' --resume %s' % file +\
                            ' --schedule 40 60' +\
                            ' --gamma 0.1' +\
                            ' --epochs %d' % args.epochs +\
                            ' --checkpoint %s' % save +\
                            ' --train-batch %d' % args.train_batch +\
                            ' --dataset %s' % args.dataset +\
                            ' --grouping_dir %s' % args.resume +\
                            ' --pruned' +\
                            ' --gpu_id %d' % (i % args.num_gpus)
            process_list[i % args.num_gpus]  = sp.Popen(exec_cmd, shell=True)
    elif args.dataset in 'imagenet':
        for i, file in enumerate(files):
            if process_list[i % args.num_gpus]:
                process_list[i % args.num_gpus].wait()
            exec_cmd = 'python3 imagenet_official_retrain.py' +\
                            ' --data %s' % args.data +\
                            ' --arch %s' % args.arch +\
                            ' --resume %s' % file +\
                            ' --schedule 10 15' +\
                            ' --config %s' % args.resume + '/grouping_config.npy' +\
                            ' --gamma 0.1 ' +\
                            ' --batch_size %d' % args.train_batch +\
                            ' --epochs %d' % args.epochs +\
                            ' --checkpoint %s' % save +\
                            ' --gpu %s' % (i % args.num_gpus)
            process_list[i % args.num_gpus]  = sp.Popen(exec_cmd, shell=True)
        
if __name__ == '__main__':
    main()
