import torch.utils.data as data

from PIL import Image
import os
import os.path

import random

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx, group = None, target_abs_index = None):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
       # pdb.set_trace()
        if target not in class_to_idx:
            continue
        if int(class_to_idx[target]) not in group:
            continue

        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    if target_abs_index != None :
                        item = (path, target_abs_index)
                    else:
                        item = (path, class_to_idx[target])
                    images.append(item)

    return images # random.sample(images, 5000) # Used for debug

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, activations = False, group = None, retrain = False):
        classes, class_to_idx = find_classes(root)

        # Case: Evaluate but pull from training set
        if activations and group:
            imgs = make_dataset(root, class_to_idx, group)
        elif group is not None: # Case: Train / Evaluate: pos/neg according to group
             if retrain: # Subcase: Retraining (Training Set Creation)
                imgs = []
                for abs_index, class_index in enumerate(group):
                    pos_imgs = make_dataset(root, \
                                            class_to_idx, \
                                            group=[class_index], \
                                            target_abs_index=abs_index + 1)
                    multiplier = max(1, 0) # Multiple used to balance, if wanted
                    imgs.extend(pos_imgs)
                negative_numbers = len(imgs)
                negative_indices = [i for i in range(1000) if i not in group]
                neg_imgs = make_dataset(root, \
                                        class_to_idx, \
                                        group=negative_indices, \
                                        target_abs_index=0)
                neg_imgs = random.sample(neg_imgs, negative_numbers)
                imgs.extend(neg_imgs)
                print("Num images in training set: {}".format(len(imgs)))
                # print("Added {} positive images with target index {}".format(len(pos_imgs)*multiplier, abs_index))      
             else: # Subcase: Evaluation (Validation Set Creation)
                 imgs = []
                 for abs_index, class_index in enumerate(group):
                      pos_imgs = make_dataset(root, \
                                              class_to_idx, \
                                              group=[class_index], \
                                              target_abs_index=abs_index + 1)
                      imgs.extend(pos_imgs)
                 negative_numbers = len(imgs)
                 print("positive images in val loader: ", negative_numbers)
                 negative_indices = [i for i in range(1000) if i not in group]
                 neg_imgs = make_dataset(root, \
                                         class_to_idx, \
                                         group=negative_indices, \
                                         target_abs_index=0)
                 
                 neg_imgs = random.sample(neg_imgs, negative_numbers)
                 imgs.extend(neg_imgs)
                 print("Num images in validation set {}".format(len(imgs)))
        else: # Case: Default
             imgs = make_dataset(root, class_to_idx, group = [i for i in range(1000)])
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)



