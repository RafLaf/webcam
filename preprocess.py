"""
operations relevent for input/output of the neural network and the rest of the code

"""
import torch
from torchvision import transforms
#import torch.nn.functional as F
import numpy as np




def image_preprocess(img):
    """
    preprocess a given image into a Tensor (rescaled and center crop + normalized)
        Args :
            img(PIL Image or numpy.ndarray): Image to be prepocess.
        returns :
            img(torch.Tensor) : preprocessed Image
    """
    img = transforms.ToTensor()(img)
    norm = transforms.Normalize(
        np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
        np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]),
    )
    all_transforms = torch.nn.Sequential(
        transforms.Resize(110), transforms.CenterCrop(100), norm
    )
    img = all_transforms(img)
    return img


def feature_preprocess(features, mean_base_features):
    """
    preprocess the feature (normalisation on the unit sphere) for classification
        Args :
            features(torch.Tensor) : feature to be preprocessed
            mean_base_features(torch.Tensor) : expected mean of the tensor
        returns:
            features(torch.Tensor) : normalized feature
    """
    features = features - mean_base_features
    features = features / torch.norm(features, dim=1, keepdim=True)
    return features
