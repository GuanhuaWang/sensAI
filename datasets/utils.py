from typing import Tuple
import numpy as np
import math


class DataSetWrapper(object):
    def __init__(self, dataset, class_group: Tuple[int], negative_samples=False):
        # The original dataset has been shuffled. Skip shuffling this dataset
        # for consistency.
        self.dataset = dataset
        self.class_group = class_group
        self.negative_samples = negative_samples
        self.targets = np.asarray(self.dataset.targets)
        # This is the bool mask for all classes in the given group.
        positive_mask = np.zeros_like(self.targets, dtype=bool)
        for class_index in class_group:
            positive_mask |= (self.targets == class_index)
        positive_class_indices = np.where(positive_mask)[0]
        if negative_samples:
            # For N negative samples, P positive samples, we need to append
            # (k * N - P) positive samples.
            k = len(class_group)
            P = len(positive_class_indices)
            N = len(self.targets) - P
            assert N >= P, "there are already more positive classes"
            ext_P = k * N - P
            repeat_n = math.ceil(ext_P / P)
            extented_indices = np.repeat(
                positive_class_indices, repeat_n)[:ext_P]
            # fuse and shuffle
            all_indices = np.arange(len(self.targets))
            fullset = np.concatenate([all_indices, extented_indices])
            np.random.shuffle(fullset)
            self.mapping = fullset
        else:
            self.mapping = positive_class_indices

    def __getitem__(self, i):
        index = self.mapping[i]
        data, label = self.dataset[index]
        if label in self.class_group:
            label = list(self.class_group).index(label) + 1
        else:
            label = 0
        return data, label

    def __len__(self):
        return len(self.mapping)

    @property
    def num_classes(self):
        return len(self.class_group) + 1
