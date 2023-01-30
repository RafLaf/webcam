
from backbone_loader.backbone.resnet12 import ResNet12

from backbone_loader.backbone_loader import load_model_weights
import torch 


backbone_type="cifar_tiny"

BACKBONE_SPECS={
    "type":"pytorch",
    "device":"cuda:0",
    "model_name": "easy-resnet12",  
    "kwargs": {
        "input_shape": [3, 32, 32],
        "num_classes": 64,  # 351,
        "few_shot": True,
        "rotations": False,
    },

}
if backbone_type=="cifar_small":
    BACKBONE_SPECS["weight"]="weight/smallcifar1.pt1"
    BACKBONE_SPECS["kwargs"]["feature_maps"]=45

elif backbone_type=="cifar":
    BACKBONE_SPECS["weight"]="weight/cifar1.pt1"
    
    BACKBONE_SPECS["kwargs"]["feature_maps"]=64

elif backbone_type=="cifar_tiny":
    BACKBONE_SPECS["weight"]="weight/tinycifar1.pt1"
    BACKBONE_SPECS["kwargs"]["feature_maps"]=32
    print(BACKBONE_SPECS)



output_names = [ "Output" ] 

_,h,w=BACKBONE_SPECS["kwargs"]["input_shape"]
feature_map=BACKBONE_SPECS["kwargs"]["feature_maps"]
name_onnx=f"resnet12_{h}_{w}_{feature_map}.onnx" 
batch_size=1 
model=ResNet12(**BACKBONE_SPECS["kwargs"])
load_model_weights(model,BACKBONE_SPECS["weight"],device=BACKBONE_SPECS["device"])
dummy_input = torch.randn(batch_size, 3, 32,32, device="cpu") 
torch.onnx.export(model, dummy_input, name_onnx, verbose=True, opset_version=10, output_names=output_names)