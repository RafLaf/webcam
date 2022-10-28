import torch
from torchvision import transforms, datasets
import torch.nn.functional as F
import numpy as np


# Apply transformations
#TODO : transoform them to numpy arrray for compatibility with pynk

def image_preprocess(img):
    img = transforms.ToTensor()(img)
    norm = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    all_transforms = torch.nn.Sequential(transforms.Resize(110), transforms.CenterCrop(100), norm)
    img = all_transforms(img)
    return img

def feature_preprocess(features, mean_base_features=None):
    features = features - mean_base_features
    features = features / torch.norm(features, dim = 1, keepdim = True)
    return features