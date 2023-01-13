"""
get the backbone with the specified framework, using argument from 

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
        from backbone_loader.backbone_loader_pytorch import TorchBatchModelWrapper

        device=model_specs["device"]
        model_name=model_specs["model_name"]
        path_weight=model_specs["path"]
        kwargs=model_specs["kwargs"]
        return TorchBatchModelWrapper(device,model_name,path_weight,kwargs)
    elif model_specs["type"]=="tensil_model":
        from backbone_loader.backbone_tensil import backbone_tensil_wrapper
        
        return backbone_tensil_wrapper(model_specs["path_bit"],model_specs["path_tmodel"])
    elif model_specs["type"]=="onnx":
        from backbone_loader.backbone_onnx import backbone_onnx_wrapper
        return backbone_onnx_wrapper(model_specs["path_onnx"])

    else:
        raise UserWarning("model type="+model_specs["type"]+"is not defined")
