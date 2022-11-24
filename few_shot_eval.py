"""
author : ybendou
function to evaluatea a dataset in the few shot setting

"""

import torch
import numpy as np
from args import *
from utils import *

n_runs = args.n_runs
batch_few_shot_runs = args.batch_fs
assert(n_runs % batch_few_shot_runs == 0)

def define_runs(n_runs,n_ways, n_shots, n_queries, num_classes, elements_per_class):

    """
    Define a few shot run setting

    args :
        - n_ways : number of classe to infer 
        - n_shots : number of shot for each class
        - n_queries : number of image to infer per class
        - num_classes : total number of class in the dataset
        - element_per_class(int) : total number of element per class in the dataset

    return: 
    list(
        - run_classes(torch.tensor(n_runs,n_ways)) :
            classe of the various images 
        - run_indices(torch.tensor(n_runs,n_ways,n_shots+n_queries)) :
            wich image in the dataset 
    ) for each n_runs in arg.n_runs
    
    """
   
    run_classes = torch.LongTensor(n_runs, n_ways).to(args.device)
    run_indices = torch.LongTensor(n_runs, n_ways, n_shots + n_queries).to(args.device)
    for i in range(n_runs):
        run_classes[i] = torch.randperm(num_classes)[:n_ways]
        for j in range(n_ways):
            run_indices[i,j] = torch.randperm(elements_per_class[run_classes[i, j]])[:n_shots + n_queries]
    return run_classes, run_indices

def define_runs_from_list(n_runs,n_ways, n_shots_list, n_queries, num_classes, elements_per_class):
    """
    return a list of tuple (dim 1 = classe, dim2= indices). Each indice of the tuple correspond to a different number of shots
    """

    return list(zip(*[define_runs(n_runs,n_ways, s, n_queries, num_classes, elements_per_class) for s in n_shots_list]))


def get_features(model, loader, n_aug = 1):
    """
    get the features given by a model
    adapted from the Easy repo to work with numpy array
    - args :
        - model(callable) : inference backbone that return  features
        - loader : iterator yielding batch of image/target (proprocessed)
    """

    
    for augs in range(n_aug):
        all_features, offset, max_offset = [], 1000000, 0
        for batch_idx, (data, target) in enumerate(loader):        
            features = model(data)
            all_features.append(features)
            offset = min(min(target), offset)
            max_offset = max(max(target), max_offset)
        num_classes = max_offset - offset + 1

        if augs == 0:
            features_total = np.concatenate(all_features, axis = 0).reshape(num_classes, -1, all_features[0].shape[1])
        else:
            features_total += np.concatenate(all_features, dim = 0).reshape(num_classes, -1, all_features[0].shape[1])
    return features_total / n_aug

