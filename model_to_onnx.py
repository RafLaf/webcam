
"""
script will load models, generate torchinfos and onnx (simplified version) for each resolution

python model_to_onnx.py --input-resolution 32 64 128  --model-type "easy_resnet12_tiny_cifar" --model-specification "weight/tinycifar1.pt1" --weight-description "weight from easy repo (tinycifar1.pt1)" 
python model_to_onnx.py --input-resolution 32 64 128 164 --model-type "easy_resnet12_small_cifar" --model-specification "weight/smallcifar1.pt1" --weight-description "weight from easy repo (smallcifar1.pt1)" 
python model_to_onnx.py --input-resolution 32 64 128 164 --model-type "easy_resnet12_cifar" --model-specification "weight/cifar1.pt1" --weight-description "weight from easy repo (cifar1.pt1)" 

python model_to_onnx.py --input-resolution 32 64 128 200 256 --model-type "mobilenet_v2" --model-specification "pretrained" --weight-description "weight trained on imagenet"  --from-hub
python model_to_onnx.py --input-resolution  32 64 128 256 --model-type "mnasnet0_5" --model-specification "random_init" --weight-description "weight random"  --from-hub
python model_to_onnx.py --input-resolution 32 64 128 256 --model-type "nvidia_efficientnet_b0" --model-specification "random_init" --weight-description "weight random"  --from-hub
python model_to_onnx.py --input-resolution 32 64 128 256 --model-type "shufflenet_v2_x0_5" --model-specification "pretrained" --weight-description "weight trained on imagenet"  --from-hub
python model_to_onnx.py --input-resolution 32 64 128 256 --model-type "squeezenet1_1" --model-specification "pretrained" --weight-description "weight trained on imagenet"  --from-hub
python model_to_onnx.py --input-resolution 32 64 128 256 --model-type "nvidia_gpunet" --model-specification "pretrained" --weight-description "weight trained on imagenet"  --from-hub


available models:

    *mobilenet_v2
    mobilenet_v3_small
    mobilenet_v3_large
    *mnasnet0_5
    mnasnet0_75 (pretrained weight not available)
    mnasnet1_0
    *squeezenet1_1
    shufflenet_v2_x0_5
    shufflenet_v2_x1_0
    nvidia_efficientnet_b0
    *nvidia_gpunet (verification takes a long time)

dependency :
torchvision (because using model.py)
onnx : require opset 1.5.0 
if legacy-install-failure :
https://www.pythonpool.com/error-legacy-install-failure/
"""

import argparse
import json
from pathlib import Path
import os
import torchinfo
import onnx
from onnxsim import simplify
import torch
import time
import numpy as np
from tqdm import tqdm

from backbone_loader.backbone_pytorch.model import get_model
from backbone_loader.backbone_loader_pytorch import TorchBatchModelWrapper

def save_weight_description(path_description,model_specification, weight_desc):
    # Load the existing data from the JSON file
    try:
        with open(path_description) as f:
            data = json.load(f)
    except:
        data = {}
    
    # Check if the weight path is already in the data
    if model_specification not in data:
        # Append the weight description to the data
        data[model_specification] = weight_desc
        with open(path_description, "w") as f:
            json.dump(data, f)
    else:
        print("description already present, no modification will be done")

def replace_reduce_mean(onnx_model):
    """
    Replace all reduce_mean operation with GlobalAveragePool if they act on the last dimentions of the tensor
    do not change the name of this operation
    Suppose that attribute has the following element : 
        name: "axes"
        ints: -2 /2
        ints: -1 /3
        type: INTS
    """
    for node in onnx_model.graph.node:
    
        if node.name.find("ReduceMean")>=0:


            #check if node operate on the last op
            do_replace_mean=False
            number_attribute=len(node.attribute)
            for i in range(number_attribute):
                attribute=node.attribute[i]
            
                if attribute.name=="axes":
                    x,y=attribute.ints
                    if (x==2 and y==3) or (x==3 and y==2):
                        do_replace_mean=True

                    if (x==-2 and y==-1) or (x==-1 and y==-2):
                        do_replace_mean=True

            #replace the node if needed
            if do_replace_mean:
                print("Replacing ReduceMean operation with GlobalAveragePool")
                
                for i in range(number_attribute):
                    node.attribute.pop()
                node.op_type="GlobalAveragePool"
    return onnx_model


def model_to_onnx(args):
    # create model path
    # one model = sevral possible resolutions 
    # generate summary and save it 

    # args arguments
    input_resolution=[int(i) for i in args.input_resolution]
   
    if args.from_hub:
        
        model_name=f"hub_{args.model_type}_{args.model_specification}"
    else:
        model_name=f"{args.model_type}"


    # handling path
    model=get_model(args.model_type,args.model_specification)

    parent_path=Path.cwd()/"onnx"
    parent_path.mkdir(parents=False,exist_ok=True)
    info_path=parent_path/ "infos"
    info_path.mkdir(parents=False,exist_ok=True)
    weight_desc_file = info_path / f"{model_name}_description.json"

    #Saving 
    save_weight_description(weight_desc_file,args.model_specification,args.weight_description)
    
    min_res=32
    max_res=100#max(input_resolution)#TODO res is to be set in function of maximum identified resolution for fpga
    
    number_eval=100
    step_res=10
    possible_res=[i for i in range(min_res,max_res,step_res)]
    time_execution=np.zeros((len(possible_res),number_eval))

    # first emtpy call (in one occasion, i've observed that it was slower)
    dummy_input=torch.randn(1, 3, min_res,min_res, device="cpu")
    _=model(dummy_input)

    for i_res,res in tqdm(enumerate(possible_res),total=len(possible_res)):
        print("res : ",res)
        dummy_input=torch.randn(1, 3, res,res, device="cpu") 
        
        for i in range(number_eval):
            t=time.time()
            _=model(dummy_input)
            time_execution[i_res,i]=time.time()-t
    
    
    np.savez(info_path/f"exec_time_{model_name}.npy",possible_res=possible_res,time_execution=time_execution)
    
    
    print("possible resolutions : ",possible_res)
    print("mean execution time : ",np.mean(time_execution,axis=-1))
    print("mean fps : ",1/np.mean(time_execution,axis=-1))
    
    # for each input, create the corresponding file
    
    for input_resolution in input_resolution:
       
        resolution_folder = parent_path / f"{input_resolution}x{input_resolution}"
        resolution_folder.mkdir(parents=False,exist_ok=True)
        
        ans=torchinfo.summary(model,(3,input_resolution,input_resolution),batch_dim = 0,verbose=0,device="cpu",col_names=
            ["input_size",
            "output_size",
            "num_params",
            "kernel_size",
            "mult_adds"])

        
        
        
        dummy_input = torch.randn(1, 3, input_resolution,input_resolution, device="cpu") 

        
        with open(resolution_folder/ f"{model_name}_torchinfo.txt","w",encoding="utf-8") as file:
            to_write= str(ans)
            file.write(to_write)

        # generate onnx
        path_model=resolution_folder/ f"{model_name}_{input_resolution}_{input_resolution}.onnx"
        path_model_simp=resolution_folder/ f"simp_{model_name}_{input_resolution}_{input_resolution}.onnx"
        torch.onnx.export(model, dummy_input, path_model, verbose=False, opset_version=10, output_names=[args.output_names])
        
        #load onnx

        onnx_model = onnx.load(path_model)
        onnx_model=replace_reduce_mean(onnx_model)
        # convert model
        #model_simp, check = simplify(onnx_model)
        #assert check, "Simplified ONNX model could not be validated"
        #path_model_simp=resolution_folder/ f"simp-{model_name}-{input_resolution}_{input_resolution}.onnx"#if one wants to test difference
        onnx.save(onnx_model,path_model)
    
if __name__ == "__main__":
    
    # Define the command line arguments for the script
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-resolution", nargs='+', required=True, help="Input resolution(s) of the images (squared images), ex: 32 64")
    parser.add_argument("--from-hub",action='store_true',help="if true, add hub- to name")
    parser.add_argument("--model-type",type=str, required=True, help="Specification of the model")
    parser.add_argument("--model-specification", type=str, required=True, help="additional specs for model. 1. easy_resnet, path to weight 2. one of hardcoded kwargs in model.py")
    parser.add_argument("--weight-description", required=True, help="Description of the weight file")
    parser.add_argument("--output-names",default="Output", help="Name of the output layer")
    
    #parser.add_argument("--perform-evaluation",action='store_true', help="if specified, will perform inference")
    # Parse the command line arguments
    args = parser.parse_args()
    model_to_onnx(args)


    