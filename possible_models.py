"""
neural network modules
handle loading, inference and prediction
"""
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

from resnet12 import ResNet12

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




def load_model_weights(model, path, device=None):
    """
    load the weight given by the path
    if the weight is not correct, raise an errror
    if the weight is not correct, may have no loading at all
        args:
            model(torch.nn.Module) : model on wich the weights should be loaded
            path(...) : a file-like object (path of the weights)
            device(torch.device) : the device on wich the weights should be loaded (optional)
    """
    pretrained_dict = torch.load(path, map_location=device)
    model_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    new_dict = {}
    for k, weight in pretrained_dict.items():
        if k in model_dict:

            # bn : keep precision (low cost associated)
            # does this work for the fpga ?
            if "bn" in k:
                new_dict[k] = weight
            else:
                new_dict[k] = weight.to(torch.float16)
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    print("Model loaded!")


def get_model(model, model_specs):
    """
    get the model specified in input
    returns :
        resnet(torch.nn.Module) : neural network corespounding to parameters
    """
    if model == "resnet12":
        return ResNet12(**model_specs)
    raise NotImplementedError(f"model {model} is not implemented")


def predict_feature(shots_list, features, model_name, **kwargs):
    """
    TODO : change dtype to numpy array (4.)
    predict the class of a features with a model
    args:
        shots_list (list[torch.Tensor]) : previous representation for each class
        features (torch.Tensor) : feature to predcit
        model_name : wich model do we use
        **kwargs : additional parameters of the model
    returns :
        classe_prediction : class prediction
        probas : probability of belonging to each class
    """
    if model_name == "ncm":
        shots = torch.stack([s.mean(dim=0) for s in shots_list])
        distances = torch.norm(shots - features, dim=1, p=2)
        classe_prediction = distances.argmin().item()
        probas = F.softmax(-20 * distances, dim=0).detach().cpu()
    elif model_name == "knn":
        number_neighboors = kwargs["number_neighboors"]
        shots = torch.cat(shots_list)
        # create target list of the shots
        targets = torch.cat(
            [torch.Tensor([i] * shots_list[i].shape[0]) for i in range(len(shots_list))]
        )
        distances = torch.norm(shots - features, dim=1, p=2)
        # get the k nearest neighbors

        _, indices = distances.topk(number_neighboors, largest=False)
        probas = (
            F.one_hot(
                targets[indices].to(torch.int64), num_classes=len(shots_list)
            ).sum(dim=0)
            / number_neighboors
        )
        classe_prediction = probas.argmax().item()
    return classe_prediction, probas

def get_features(img,backbone,device):
    """
    wrapper for the model
    TODO : return a numpy array (3.)
    args :
        img(PIL Image or numpy.ndarray) : current img
        backbone(torch.nn.Module) : neural network that will output features
        device(torch.device) : the device on wich the weights should be loaded 
        TODO : replace device argument by backbone.device
    returns : 
        features : preprocessed featured of img
    """
    img = image_preprocess(img).to(device)
    _, features = backbone(img.unsqueeze(0))
    return features

def feature_preprocess(features, mean_base_features):
    """
    TODO : change dtype to numpy array (2.)
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



def get_preprocessed_feature(img,backbone,mean_features,device):
    """
    compute the normalized feature for the given img
    args :
        img(PIL Image or numpy.ndarray) : current img
        backbone(torch.nn.Module) : neural network that will output features
        mean_features (torch.Tensor) : mean of all features
        device(torch.device) : the device on wich the weights should be loaded
    returns : 
        features : preprocessed featured of img
    """
    features=get_features(img,backbone,device)
    features = feature_preprocess(features, mean_features)
    return features

def predict_class_moving_avg(img, shots_list, mean_features, backbone, classifier_specs, prev_probabilities,device):
    """
    TODO : change dtype to numpy array (1.)
    update the probabily and attribution of having a class, using the current image
    args :
        img(PIL Image or numpy.ndarray) : current img,
        shots_list (list[torch.Tensor]) : previous representation for each class
        mean_features (torch.Tensor) : mean of all features
        backbone(torch.nn.Module) : neural network that will output features
        classifier_specs(dict) : specification of the classifier
        prev_probabilities(?) : probability of each class
    returns :
        classe_prediction : class prediction
        probas : probability of belonging to each class
    """
    features=get_preprocessed_feature(img,backbone,mean_features,device)

    model_name = classifier_specs["model_name"]
    _, probas = predict_feature(
        shots_list, features, model_name, **classifier_specs["args"]
    )
    print("probabilities:", probas)

    if prev_probabilities is None:
        prev_probabilities = probas
    else:
        if model_name == "ncm":
            probabilities = prev_probabilities * 0.85 + probas * 0.15
        elif model_name == "knn":
            probabilities = prev_probabilities * 0.95 + probas * 0.05

    classe_prediction = probabilities.argmax().item()
    return classe_prediction, probabilities
