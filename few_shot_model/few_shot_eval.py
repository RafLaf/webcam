"""
author : ybendou
function to evaluate a dataset in the few shot setting

"""

import numpy as np
from args import args
import tqdm
from few_shot_model.numpy_utils import *

n_runs = args.n_runs
batch_few_shot_runs = args.batch_size
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
    rng = np.random.default_rng()
    run_classes = np.zeros((n_runs,n_ways),dtype=np.int64)#torch.LongTensor(n_runs, n_ways).to(args.device)
    run_indices = np.zeros((n_runs,n_ways,n_shots+n_queries),dtype=np.int64)#torch.LongTensor(n_runs, n_ways, n_shots + n_queries).to(args.device)
    for i in range(n_runs):
        run_classes[i] = rng.permutation(num_classes)[:n_ways]#torch.randperm(num_classes)[:n_ways]
        for j in range(n_ways):
            run_indices[i,j] = rng.permutation(elements_per_class[run_classes[i, j]])[:n_shots + n_queries] #torch.randperm(elements_per_class[run_classes[i, j]])[:n_shots + n_queries]
    return run_classes, run_indices

def define_runs_from_list(n_runs,n_ways, n_shots_list, n_queries, num_classes, elements_per_class):
    """
    return a list of tuple (dim 1 = classe, dim2= indices). Each indice of the tuple correspond to a different number of shots
    (currently not used)
    """

    return list(zip(*[define_runs(n_runs,n_ways, s, n_queries, num_classes, elements_per_class) for s in n_shots_list]))


def get_features_few_shot_ds_pytorch(model, loader, n_aug = 1):
    """
    get the features given by a model
    adapted from the Easy repo to work with numpy array
    Be carful, the classes are supposed to be ordered.
    - args :
        - model(callable) : inference backbone that return  features
        - loader : iterator yielding batch of image/target (proprocessed)
    """

    
    for augs in range(n_aug):
        all_features, offset, max_offset = [], 1000000, 0
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(loader)):        
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


def get_features_numpy(model, data,batch_size):
    """
    get the features given by a model
    adapted from the Easy repo to work with numpy array
    - args :
        - model(callable) : inference backbone that return batch of features (supposed to return numpy ndarray)
        - data(np.ndarray (num_classes,num_batch,height,width,channel)) : numpy array contening the data
        - batch_size : how many image for one forward pass
    """
    (num_classes,num_img,height,width,channel)=data.shape
   
    empty_output=model(np.zeros((1,height,width,channel),dtype=data.dtype))#perform empty forward to get number of filter and type
    
    total_filter=np.zeros((num_classes,num_img,empty_output.shape[-1]),empty_output.dtype)
    for classe in range(num_classes):
        number_full_batch=num_img//batch_size
        for batch_number in tqdm.tqdm(range(number_full_batch)):

            data_sample=data[classe,batch_size*batch_number:batch_size*(batch_number+1)]
            output=model(data_sample)

            total_filter[classe,batch_size*batch_number:batch_size*(batch_number+1)]=output
        #last incomplete batch if needed
        if num_img%batch_size!=0:
            data_sample=data[classe,batch_size*number_full_batch:]
            output=model(data_sample)
            total_filter[classe,batch_size*number_full_batch:]=output

    return total_filter
