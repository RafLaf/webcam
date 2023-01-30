import itertools
import numpy as np
import pickle




def get_dataset_numpy(dataset_path,dtype=np.float32,number_sample_per_class=1000,dim_img=(3,32,32)):
    """
    load a few shot type dataset with pickle and return it 
    as a numpy array. Channel first type is supposed

    Args :
        dataset_path
        dtype : output dtype (has nothing to do with the way the images are incoded)
        number_sample_per_class : number of image per class (will raise an error if it does not)
        dim_img : the dimention of the encoded image. Be carrful, channel first only
    Returns:
        
        np.ndarray (n_class,number_sample_per_class,dim_img[1],dim_img[2],dim_img[3]) :
            dataset with channel last convention
    """

    # encoding type from
    # https://www.binarystudy.com/2021/09/how-to-load-preprocess-visualize-CIFAR-10-and-CIFAR-100.html
    print("opening dataset :",dataset_path)
    with open(dataset_path, 'rb') as fo:
        d = pickle.load(fo, encoding='latin1')

    iterator=zip(d["labels"],d["data"])

    sorted_iter=sorted(iterator,key=lambda d:d[0])
    # itertools need a sorted array
    # https://stackoverflow.com/questions/8116666/itertools-groupby-not-grouping-correctly

    grouped=itertools.groupby(sorted_iter,lambda d:d[0])

    numpy_data=np.zeros((0,number_sample_per_class,dim_img[1],dim_img[2],dim_img[0]),dtype=dtype)
    size=dim_img[0]*dim_img[1]*dim_img[2]


    for key,group in grouped:
        #key_and_group = {key : list(group)}
        
        transformed_group=map(lambda d:d[1],group)#remove key
        type_iter=np.dtype((np.float16,size))#dim_img
        
        group_iterators=itertools.chain.from_iterable(transformed_group)
        
        numpy_data=np.concatenate([
            numpy_data,
            np.transpose(
                np.fromiter(group_iterators,float)
                .reshape(-1,*dim_img),
                (0,2,3,1)
            ).astype(dtype)
            [None,:]
            ])
    return numpy_data

if __name__=="__main__":
    import matplotlib.pyplot as plt
    dataset=get_dataset_numpy("data/cifar-10-batches-py/test_batch")
    plt.imshow(dataset[0][0][0]/255)
    plt.show()