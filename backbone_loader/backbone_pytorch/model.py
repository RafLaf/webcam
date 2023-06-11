"""
Define all the models.

Keys of EASY_SPECS/BRAIN_RESNET12_SPECS/BRAIN_RESNET9_SPECS : name of the implemented model
attrributes : keywords arguments passed to the corresponding model
"""
import torch
from backbone_loader.backbone_pytorch.resnet12 import ResNet12
from backbone_loader.backbone_pytorch.resnet12_brain import ResNet12Brain, ResNet9

# ------------------ BUILT - IN CONFIGS -----------------------------
EASY_SPECS = {
    "easy_resnet12_small": {
        "feature_maps": 45,
        "num_classes": 64,
    },
    "easy_resnet12": {
        "feature_maps": 64,
        "num_classes": 64,
    },
    "easy_resnet12_tiny": {
        "feature_maps": 32,
        "num_classes": 64,
    },
}


BRAIN_RESNET12_SPECS = {
    "brain_resnet12_small": {
        "feature_maps": 45,
    },
    "brain_resnet12": {
        "feature_maps": 64,
    },
    "brain_resnet12_tiny": {
        "feature_maps": 32,
    },
    "brain_resnet12_small_strided": {"feature_maps": 45, "use_strides": True},
    "brain_resnet12_strided": {
        "feature_maps": 64,
        "use_strides": True,
    },
    "brain_resnet12_tiny_strided": {
        "feature_maps": 32,
        "use_strides": True,
    },
}

BRAIN_RESNET9_SPECS = {
    "brain_resnet9_small": {
        "feature_maps": 45,
    },
    "brain_resnet9": {
        "feature_maps": 64,
    },
    "brain_resnet9_tiny": {
        "feature_maps": 32,
    },
    "brain_resnet9_small_strided": {"feature_maps": 45, "use_strides": True},
    "brain_resnet9_strided": {
        "feature_maps": 64,
        "use_strides": True,
    },
    "brain_resnet9_tiny_strided": {
        "feature_maps": 32,
        "use_strides": True,
    },
    "brain_resnet9_fm16_strided": {
        "feature_maps": 16,
        "use_strides": True,
    },
}

def load_model_weights(
    model, path, device=None, verbose=False, raise_error_incomplete=True
):
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
                print(f"loading weight name : {k}", flush=True)

            if "bn" in k:
                new_dict[k] = weight.to(torch.float16)
            else:
                new_dict[k] = weight.to(torch.float16)
        else:
            if raise_error_incomplete:
                raise TypeError("the weights does not correspond to the same model")
            print("weight with name : {k} not loaded (not in model)")
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)


def get_model(backbone, input_model, use_strides, device="cpu"):
    """
    get a model from pytorch_hub or from custom arch, using hardcoded specifications
    backbone : type of the model: either resnet 12 or resnet 9.
    input_model : path to pytorch weights
    """
    pretrained_dict = torch.load(input_model, map_location=device)
    feature_maps = len(pretrained_dict['block1.conv1.conv.weight'])

    if backbone == "resnet12":
        model = ResNet12Brain(feature_maps, use_strides).to(device)
        load_model_weights(model, input_model, device=device)
    elif backbone == "resnet9":
        model = ResNet9(feature_maps, use_strides).to(device)
        load_model_weights(model, input_model, device=device)
    else:
        raise NotImplementedError(f"model {backbone} is not implemented")
    model.eval()
    return model
