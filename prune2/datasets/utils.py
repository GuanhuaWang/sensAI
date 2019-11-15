from typing import Tuple
import numpy as np


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
            # extend number of positive instances to the number of negative ones
            positive_class_len = len(positive_class_indices)
            negative_class_len = len(self.targets) - positive_class_len
            assert negative_class_len >= positive_class_len, "there are already more positive classes"
            repeat_n = negative_class_len // positive_class_len
            extented_indices = np.repeat(positive_class_indices, repeat_n)[
                :negative_class_len-positive_class_len]
            # fuse and shuffle
            fullset = np.concatenate(
                [np.ones_like(self.targets, dtype=bool), extented_indices])
            np.random.shuffle(fullset)
            self.mapping = fullset
        else:
            self.mapping = positive_class_indices

    def __getitem__(self, i):
        index = self.mapping[i]
        data, label = self.dataset[index]
        if label in self.class_group:
            label = self.class_group.index(label) + 1
        else:
            label = 0
        return data, label

    def __len__(self):
        return len(self.mapping)

    @property
    def num_classes(self):
        return len(self.dataset.classes)
