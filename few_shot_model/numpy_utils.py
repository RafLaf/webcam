"""
Redefine some equivalent of classical torch function not implemented in numpy
"""

import numpy as np


def softmax(x: np.ndarray, dim=0):
    """
    ref : https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python

    args :
        - x (np.array(np.ndarray [...]) : array with at least dim+1 dimensions
        - dim : dim allong wich to perform softmax
    """
    # stability trick( cond of exp(x)=x )
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=dim, keepdims=True)


def one_hot(array: np.ndarray, num_classes: int, dtype=np.int64):
    """
    ref : https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-one-hot-encoded-array-in-numpy
    args :
        - array(np.ndarray)[n] : input array of indice (values from 0 to num_class-1)
        - num_class: number of classes for one hot encoding

    """
    return np.identity(num_classes, dtype)[array]


def k_small(distance: np.ndarray, number: int, axis=-1):
    """
    ref : https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
    args:
        - distance :
        - number : number of indices to outputs
        - axis :  on wich axis should we look for the k smallest values

    """
    semi_sorted_dist = np.argpartition(distance, number, axis=axis)
    return np.take(semi_sorted_dist, np.arange(0, number), axis=axis)
