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