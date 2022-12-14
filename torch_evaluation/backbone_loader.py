"""
Contains the backbone (neural net) of the model. 

In embedded setting, we want to use another framework than pytorch to make inference. 

-> input : numpy or torch tensor img (preprocessed)
-> output : numpy img
"""


        
    

def get_model(model_specs):
    """
    get the model specified in input
    args :
        - model_specs
        - device
    returns :
        resnet(torch.nn.Module) : 
        neural network corespounding to parameters
            takes a batch 
    """
    
    if model_specs["type"]=="pytorch_batch":
        from torch_evaluation.backbone_loader_pytorch import TorchBatchModelWrapper

        device=model_specs["device"]
        model_name=model_specs["model_name"]
        path_weight=model_specs["path"]
        kwargs=model_specs["kwargs"]
        return TorchBatchModelWrapper(device,model_name,path_weight,kwargs)
    else:
        raise UserWarning("model type="+model_specs["type"]+"is not defined")
