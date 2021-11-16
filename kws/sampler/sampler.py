import numpy as np

import torch
from torch.utils.data import WeightedRandomSampler


def get_sampler(target):
    """
    Sampler for oversampling.

    We should provide to WeightedRandomSampler _weight for every sample_; by default it is 1/len(target).
    """
    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])  # for every class count it's number of occ.
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.float()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler
