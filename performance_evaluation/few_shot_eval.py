"""
author : adapted from ybendou in order to use numpy instead of pytorch
function to evaluate a dataset in the few shot setting

"""

import numpy as np
import tqdm
from few_shot_model.numpy_utils import *


def define_runs(
    n_runs: int,
    n_ways: int,
    n_shots: int,
    n_queries: int,
    num_classes: int,
    elements_per_class: int,
):
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
    run_classes = np.zeros((n_runs, n_ways), dtype=np.int64)
    run_indices = np.zeros((n_runs, n_ways, n_shots + n_queries), dtype=np.int64)
    for i in range(n_runs):
        run_classes[i] = rng.permutation(num_classes)[
            :n_ways
        ]  # torch.randperm(num_classes)[:n_ways]
        for j in range(n_ways):
            run_indices[i, j] = rng.permutation(elements_per_class[run_classes[i, j]])[
                : n_shots + n_queries
            ]  # torch.randperm(elements_per_class[run_classes[i, j]])[:n_shots + n_queries]
    return run_classes, run_indices


def get_features_numpy(model, data: np.ndarray, batch_size: int):
    """
    get the features given by a model
    adapted from the Easy repo to work with numpy array
    - args :
        - model(callable) : inference backbone that return batch of features (supposed to return numpy ndarray)
        - data(np.ndarray (num_classes,num_batch,height,width,channel)) : numpy array contening the data
        - batch_size : how many image for one forward pass
    """
    (num_classes, num_img, height, width, channel) = data.shape

    empty_output = model(
        np.zeros((1, height, width, channel), dtype=data.dtype)
    )  # perform empty forward to get number of filter and type

    total_filter = np.zeros(
        (num_classes, num_img, empty_output.shape[-1]), empty_output.dtype
    )
    for classe in range(num_classes):
        number_full_batch = num_img // batch_size
        for batch_number in tqdm.tqdm(range(number_full_batch)):
            data_sample = data[
                classe, batch_size * batch_number : batch_size * (batch_number + 1)
            ]
            output = model(data_sample)

            total_filter[
                classe, batch_size * batch_number : batch_size * (batch_number + 1)
            ] = output
        # last incomplete batch if needed
        if num_img % batch_size != 0:
            data_sample = data[classe, batch_size * number_full_batch :]
            output = model(data_sample)
            total_filter[classe, batch_size * number_full_batch :] = output

    return total_filter
