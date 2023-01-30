import torch
from backbone_loader.backbone_pytorch.resnet12 import ResNet12


MODEL_LOC={
    "mobilenet_v2":"pytorch/vision:v0.10.0",
    "mobilenet_v3_small":"pytorch/vision:v0.10.0",
    "mobilenet_v3_large":"pytorch/vision:v0.10.0",
    "mnasnet0_5":"pytorch/vision:v0.10.0",
    "mnasnet0_75":"pytorch/vision:v0.10.0",
    "mnasnet1_0":"pytorch/vision:v0.10.0",
    "squeezenet1_1":"pytorch/vision:v0.10.0",
    "shufflenet_v2_x0_5":"pytorch/vision:v0.10.0",
    "shufflenet_v2_x1_0":"pytorch/vision:v0.10.0",
    "efficientnet_b0":"pytorch/vision:v0.10.0",
    "nvidia_efficientnet_b0":"NVIDIA/DeepLearningExamples:torchhub",
    "nvidia_gpunet":"NVIDIA/DeepLearningExamples:torchhub"}


#for pytorch vision, supported for v>=0.13
MODEL_WEIGHT={
    "mobilenet_v2_imagenet":"MobileNet_V2_Weights.IMAGENET1K_V2",
    "mobilenet_v3_small_imagenet":"MobileNet_V3_Small_Weights.IMAGENET1K_V1",
    "mobilenet_v3_large_imagenet":"MobileNet_V3_Large_Weights.IMAGENET1K_V2",
    "mnasnet0_5_imagenet":"MNASNet0_5_Weights.IMAGENET1K_V1",
    "mnasnet0_75_imagenet":"MNASNet0_75_Weights.IMAGENET1K_V1",
    "mnasnet1_0_imagenet":"MNASNet1_0_Weights.IMAGENET1K_V1"
}


EASY_SPECS={
    "easy-resnet12-small-cifar":{
        "feature_maps":45, 
        "num_classes":64, 

    },
    "easy-resnet12-cifar":{
        "feature_maps":64, 
        "num_classes":64, 

    },
    "easy-resnet12-tiny-cifar":{
        "feature_maps":32, 
        "num_classes":64, 

    }
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

            # bn : keep precision (low cost associated)
            # does this work for the fpga ?
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
    print("Model loaded!")


def load_model_pytorch_hub(model_name,weights,device="cpu"):
    """

        load a model. currently only pytorch-hub keyword supported : pretrained and weights

        weights(str) : if "pretrained"/"random-init", used pretrained=True/pretrained=False, 
                    else if "weight", weights="weights when calling pytorch model

    """

    if weights=="pretrained":
        model=torch.hub.load(MODEL_LOC[model_name],model_name,pretrained=True)
    elif weights=="random-init":
        model=torch.hub.load(MODEL_LOC[model_name],model_name,pretrained=False)
    else:
        weights=MODEL_WEIGHT[f"{model_name}_{weights}"]
        model=torch.hub.load(MODEL_LOC[model_name],model_name,weights=weights)
    model.to(device)


    return model




def get_model(model_name,weight,device="cpu"):
    
    if model_name.find("easy-resnet12")>=0:#if str contains the model

        model = ResNet12(**EASY_SPECS[model_name]).to(device)
        load_model_weights(model, weight, device=device)
    elif model_name in MODEL_LOC.keys():
        return load_model_pytorch_hub(model_name,weight,device=device)
    else:
        raise NotImplementedError(f"model {model_name} is not implemented")
    return model