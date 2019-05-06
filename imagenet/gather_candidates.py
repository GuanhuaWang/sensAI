import subprocess as sp
import numpy as np

def main():
    # simple naive grouping, in order
    num_gpus = 4
    num_groups = 10
    group_size = 1000 // num_groups
    groups = [[i for i in range((j) * group_size, (j+1) * group_size)] for j in range(num_groups) ]
    print("%s" % (' '.join(str(x) for x in groups[0])))
    process_list = [None for _ in range(num_gpus)]
    for i, group in enumerate(groups):
         if process_list[i % num_gpus]:
             process_list[i % num_gpus].wait()
         exec_cmd = 'python3 imagenet_activations.py' +\
                     ' /home/ubuntu/imagenet --gpu %d' % (i % num_gpus) + \
                     ' --arch vgg19_bn --evaluate --pretrained --group %s' % (' '.join(str(digit) for digit in group)) + \
                     ' --name %s' % (str(i))
         process_list[i % num_gpus]  = sp.Popen(exec_cmd, shell=True)
    
    # Save the grouping class index partition information
    np.save(open("prune_candidate_logs/grouping_config.npy", "wb"), groups)


if __name__ == '__main__':
    main()
