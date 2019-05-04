import subprocess as sp


def main():
    # simple naive grouping, in order
    num_gpus = 4
    num_groups = 10
    group_size = 1000 // num_groups
    groups = [[i for i in range((j) * group_size, (j+1) * group_size)] for j in range(num_groups) ]
    print("%s" % (' '.join(str(x) for x in groups[0])))
    process_list = [None for _ in range(num_gpus)]
    for i, group in enumerate(groups):
         if process_list[i % 4]:
             process_list[i % 4].wait()
         exec_cmd = 'python3 imagenet_activations.py' +\
                     ' /home/ubuntu/imagenet --gpu %d' % (i % 4) + \
                     ' --arch vgg19_bn --evaluate --pretrained --group %s' % (' '.join(str(digit) for digit in group)) + \
                     ' --name %s' % (str(i))
         process_list[i % 4]  = sp.Popen(exec_cmd, shell=True)

if __name__ == '__main__':
    main()
