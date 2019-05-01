import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Transformations
RC   = transforms.RandomCrop(32, padding=4)
RHF  = transforms.RandomHorizontalFlip()
RVF  = transforms.RandomVerticalFlip()
NRM  = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
TT   = transforms.ToTensor()
TPIL = transforms.ToPILImage()

# Transforms object for trainset with augmentation
transform_with_aug = transforms.Compose([TPIL, RC, RHF, TT, NRM])
# Transforms object for testset with NO augmentation
transform_no_aug   = transforms.Compose([TT, NRM])

# Representative images index
# repre_idx = np.loadtxt('bottom500.txt')
# print(repre_idx.shape)
# repre_top_idx = np.loadtxt('top500.txt')
# print(repre_top_idx.shape)
# Downloading/Louding CIFAR10 data
trainset  = CIFAR10(root='./data', train=True , download=True)#, transform = transform_with_aug)
testset   = CIFAR10(root='./data', train=False, download=True)#, transform = transform_no_aug)
classDict = {'plane':0, 'car':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}

# Separating trainset/testset data/label
x_train  = trainset.train_data
x_test   = testset.test_data
y_train  = trainset.train_labels
y_test   = testset.test_labels

# Define a function to separate CIFAR classes by class index

def get_class_i(x, y, i):
    """
    x: trainset.train_data or testset.test_data
    y: trainset.train_labels or testset.test_labels
    i: class label, a number between 0 to 9
    return: x_i
    """
    # Convert to a numpy array
    y = np.array(y)
    # Locate position of labels that equal to i
    pos_i = np.argwhere(y == i)
    # Convert the result into a 1-D list
    pos_i = list(pos_i[:,0])
    # Collect all data that match the desired label
    print("positive images: %i" %(len(pos_i)))
    x_i = [x[j] for j in pos_i]
    # Debug info
    # print("x_i ", x_i)
    return x_i

def get_represent_class_i(x, y, i):
    pos_i = [int(x) for x in repre_idx[i,:]]
    pos_i = list(pos_i)
#    print(pos_i)
    x_i = [x[j] for j in pos_i]
    x_i = x_i*10
    return x_i

def get_represent_top_class_i(x, y, i):
    pos_i = [int(x) for x in repre_top_idx[i,:]]
    pos_i = list(pos_i)
#    print(pos_i)
    x_i = [x[j] for j in pos_i]
    x_i = x_i*10
    return x_i

def get_represent_neg_class_i(x, y, i):
    whole = repre_top_idx.ravel()
    pos_i = [int(x) for x in whole if x not in repre_top_idx[i,:]]
    x_i1 = [x[j] for j in pos_i]
    print(len(x_i1))

    #add another 500 random images
    y = np.array(y)
    cla_i = np.argwhere(y==i)
    whole_y = np.arange(y.size)
    np.random.shuffle(whole_y)
    x_i2 = [x[j] for j in whole_y if j not in cla_i[:,0]]
    x_i2_500 = x_i2[:500]

    x_i = x_i1+x_i2_500
    print(len(x_i))

    return x_i

# Get random images excluding images in the target class i
def get_random_images(x, y, *indices):
 
    # Convert y to np array
    y = np.array(y)
    # Locate position of labels that equal to i
    exclude_pos = np.concatenate([np.argwhere(y == i) for i in indices]) 
    # Debug info
    print("group "+ str(indices) + " size is ", exclude_pos.size)

    # Shuffle total number of train images labels
    whole = np.arange(y.size)
    np.random.shuffle(whole)

    # Exclude images in class i
    x_i = [x[j] for j in whole if j not in exclude_pos[:,0]]
    # Debug info
    # x_r_i = x_i[:exclude_pos.size]
#    print("random_x_i size ", x_r_i)
    return x_i

class DatasetMaker(Dataset):
    def __init__(self, datasets, transformFunc = transform_no_aug):
        """
        datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
        """
        self.datasets = datasets
        self.lengths  = [len(d) for d in self.datasets]
        self.transformFunc = transformFunc
    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)
        img = self.datasets[class_label][index_wrt_class]
        img = self.transformFunc(img)
        return img, class_label

    def __len__(self):
        return sum(self.lengths)
    
    def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
        """
        Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
        """
        # Which class/bin does i fall into?
        accum = np.add.accumulate(bin_sizes)
        if verbose:
            print("accum =", accum)
        bin_index  = len(np.argwhere(accum <= absolute_index))
        if verbose:
            print("class_label =", bin_index)
        # Which element of the fallent class/bin does i correspond to?
        index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
        if verbose:
            print("index_wrt_class =", index_wrt_class)

        return bin_index, index_wrt_class

# ================== Usage ================== #

# Let's choose cats (class 3 of CIFAR) and dogs (class 5 of CIFAR) as trainset/testset
# cat_dog_trainset = \
#     DatasetMaker(
#         [get_represent_neg_class_i(x_train, y_train, classDict['cat']), get_random_images(x_train, y_train, classDict['cat'])],
#         transform_with_aug
#     )
# cat_dog_testset  = \
#     DatasetMaker(
#         [get_class_i(x_test , y_test , classDict['cat']), get_random_images(x_test , y_test, classDict['cat'])],
#         transform_no_aug
#     )
# kwargs = {'num_workers': 2, 'pin_memory': False}

# Create datasetLoaders from trainset and testset

if __name__ == '__main__':
    trainsetLoader   = DataLoader(cat_dog_trainset, batch_size=64, shuffle=True , **kwargs)
    testsetLoader    = DataLoader(cat_dog_testset , batch_size=64, shuffle=False, **kwargs)