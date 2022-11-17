"""
neural network modules
handle loading, inference and prediction
TODO : replace the functions with a unique class to avoid unessary parameters
"""
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

from resnet12 import ResNet12


def get_camera_preprocess():
    """
    preprocess a given image into a Tensor (rescaled and center crop + normalized)
        Args :
            img(PIL Image or numpy.ndarray): Image to be prepocess.
        returns :
            img(torch.Tensor) : preprocessed Image
    """
    norm = transforms.Normalize(
        np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
        np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]),
    )
    all_transforms =  transforms.Compose(
        [
        transforms.ToTensor(), transforms.Resize(110), transforms.CenterCrop(100), norm
    ])

    return all_transforms


def load_model_weights(model, path, device=None,verbose=False,raise_error_incomplete=True):
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
            if verbose:
                print(f"loading weight name : {k}",flush=True)

            # bn : keep precision (low cost associated)
            # does this work for the fpga ?
            if "bn" in k:
                new_dict[k] = weight
            else:
                new_dict[k] = weight.to(torch.float16)
        else:
            if raise_error_incomplete:
                raise TypeError("the weights does not correspond to the same model")
            print("weight with name : {k} not loaded (not in model)")
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    print("Model loaded!")


def get_model(model_specs, device):
    """
    get the model specified in input
    returns :
        resnet(torch.nn.Module) : neural network corespounding to parameters
    """
    name_model = model_specs["model_name"]
    if name_model == "resnet12":
        model = ResNet12(**model_specs["kwargs"]).to(device)
    else:
        raise NotImplementedError(f"model {name_model} is not implemented")
    load_model_weights(model, model_specs["path"], device=device)
    return model


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


class FewShotModel:
    """
    class defining a few shot model
        attributes :
            - backbone_specs(dict) :
                specs defining the how to load the backbone
            - classifier_specs :
                parameters of the final classification model
            - transform :
                transforms used to transform the input img from PIL/numpy
                into the input of the backbone
            - device :
                on wich device we should perform computation
    """

    def __init__(self, backbone_specs, classifier_specs, transform, device):
        self.backbone = get_model(backbone_specs,device)
        self.classifier_specs = classifier_specs
        self.transform = transform
        self.device = device

    def get_features(self, img):
        """
        wrapper for the backbone
        TODO : return a numpy array (3.)
        args :
            img(PIL Image or numpy.ndarray) : current img
            backbone(torch.nn.Module) : neural network that will output features
            device(torch.device) : the device on wich the weights should be loaded
            transform : tranformation to apply to the input. Default to camera setting
        returns :
            features : preprocessed featured of img
        """

        img = self.transform(img).to(self.device)
        _, features = self.backbone(img.unsqueeze(0))
        return features

    def predict_class(self, img, recorded_data, mean_feature):
        """
        predict the class of a features with a model
        TODO : change dtype to numpy array (4.)

        args:
            img(PIL Image or numpy.ndarray) : current img that we will predict
            recorded_data (list[torch.Tensor]) : previous representation for each class
            model_name : wich model do we use
            **kwargs : additional parameters of the model
        returns :
            classe_prediction : class prediction
            probas : probability of belonging to each class
        """
        model_name = self.classifier_specs["model_name"]
        model_arguments = self.classifier_specs["kwargs"]
        shots_list=recorded_data.get_shot_list()

        # compute the features and normalization
        features = self.get_features(img)
        
        features = feature_preprocess(features, mean_feature)

        # class asignement using the corespounding model

        if model_name == "ncm":
            shots = torch.stack([s.mean(dim=0) for s in shots_list])
            shots = feature_preprocess(shots, mean_feature)
            distances = torch.norm(shots - features, dim=1, p=2)
            classe_prediction = distances.argmin().item()
            probas = F.softmax(-20 * distances, dim=0).detach().cpu()

        elif model_name == "knn":
            number_neighboors = model_arguments["number_neighboors"]
            # create target list of the shots

            shots = torch.cat(shots_list)
            shots = feature_preprocess(shots, mean_feature)

            targets = torch.cat(
                [
                    torch.Tensor([i] * shots_list[i].shape[0])
                    for i in range(len(shots_list))
                ]
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
        else:
            raise NotImplementedError(f"classifier : {model_name} is not implemented")

        return classe_prediction, probas

    def predict_class_moving_avg(
        self, img, prev_probabilities, recorded_data, mean_features
    ):
        """
        TODO : change dtype to numpy array (1.)
        update the probabily and attribution of having a class, using the current image
        args :
            img(PIL Image or numpy.ndarray) : current img,
            prev_probabilities(?) : probability of each class for previous prediction
            shots_list (list[torch.Tensor]) : previous representation for each class
            mean_features (torch.Tensor) : mean of all features
        returns :
            classe_prediction : class prediction
            probas : probability of belonging to each class
        """
        model_name = self.classifier_specs["model_name"]
        

        _, current_proba = self.predict_class(img, recorded_data, mean_features)

        print("probabilities:", current_proba)

        if prev_probabilities is None:
            probabilities = current_proba
        else:
            if model_name == "ncm":
                probabilities = prev_probabilities * 0.85 + current_proba * 0.15
            elif model_name == "knn":
                probabilities = prev_probabilities * 0.95 + current_proba * 0.05

        classe_prediction = probabilities.argmax().item()
        return classe_prediction, probabilities
