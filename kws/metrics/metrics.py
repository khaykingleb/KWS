import torch
import numpy as np

import matplotlib as plt


def count_FA_FR(preds, labels):
    """
    FA - true: 0, model: 1
    FR - true: 1, model: 0
    """
    FA = torch.sum(preds[labels == 0])
    FR = torch.sum(labels[preds == 0])
    
    # torch.numel - returns total number of elements in tensor
    return FA.item() / torch.numel(preds), FR.item() / torch.numel(preds)


def get_auc_FA_FR(probs, labels):
    sorted_probs, _ = torch.sort(probs)
    sorted_probs = torch.cat((torch.Tensor([0]), sorted_probs, torch.Tensor([1])))
    labels = torch.cat(labels, dim=0)
        
    FAs, FRs = [], []
    for prob in sorted_probs:
        preds = (probs >= prob) * 1
        FA, FR = count_FA_FR(preds, labels)        
        FAs.append(FA)
        FRs.append(FR)
    plt.plot(FAs, FRs)
    plt.show()

    # ~ area under curve using trapezoidal rule
    return -np.trapz(FRs, x=FAs)
