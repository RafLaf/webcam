
"""
script will load models, generate torchinfos and onnx (simplified version) for each resolution
python model_to_onnx.py --input-resolution 32 64 128 256 --model-type "mnasnet0_75" --weight-name "random-init" --weight-description "weight trained on imagenet" 
python model_to_onnx.py --input-resolution 32 64 128 256 --model-type "mobilenet_v2" --weight-name "pretrained" --weight-description "weight trained on imagenet" --perform-evaluation

available models (*=tested):

    *mobilenet_v2
    mobilenet_v3_small
    mobilenet_v3_large
    mnasnet0_5
    *mnasnet0_75 (pretrained weight not available)
    mnasnet1_0
    squeezenet1_1
    shufflenet_v2_x0_5
    shufflenet_v2_x1_0
    *nvidia_efficientnet_b0
    *nvidia_gpunet

"""
import argparse
import json
from pathlib import Path
#import sys
#import subprocess
import os
import torchinfo
import onnx
from onnxsim import simplify
import torch

from backbone_loader.backbone_pytorch.model import get_model
from backbone_loader.backbone_loader_pytorch import TorchBatchModelWrapper
from few_shot_evaluation import evaluate_model

def save_weight_description(path_description,weight_name, weight_desc):
    # Load the existing data from the JSON file
    try:
        with open(path_description) as f:
            data = json.load(f)
    except:
        data = {}
    
    # Check if the weight path is already in the data
    if weight_name not in data:
        # Append the weight description to the data
        data[weight_name] = weight_desc
        with open(path_description, "w") as f:
            json.dump(data, f)
    else:
        print("description already present, no modification will be done")

def model_to_onnx(args):
    # create model path
    # one model = sevral possible resolutions 
    # generate summary and save it 
    if args.model_source=="pytorch-hub":
        model=get_model(args.model_type,args.weight_name)
        model_name=f"hub-{args.model_type}-{args.weight_name}"
    else:
        print("this model source is not implemented")
    model_folder = Path.cwd()/"onnx" / model_name
    model_folder.mkdir(parents=False,exist_ok=True)

    # Create the weight description file


    
    weight_desc_file = model_folder / "description.json"
    save_weight_description(weight_desc_file,args.weight_name,args.weight_description)

    # for each input, create the corresponding file
    for input_resolution in args.input_resolution:
        input_resolution=int(input_resolution)
        resolution_folder = model_folder / f"{input_resolution}x{input_resolution}"
        
        
        ans=torchinfo.summary(model,(3,input_resolution,input_resolution),batch_dim = 0,verbose=1,device="cpu",col_names=
            ["input_size",
            "output_size",
            "num_params",
            "kernel_size",
            "mult_adds"])

        resolution_folder.mkdir(parents=False,exist_ok=True)
        
        
        dummy_input = torch.randn(1, 3, input_resolution,input_resolution, device="cpu") 

        
        with open(resolution_folder/"torchinfo.txt","w",encoding="utf-8") as file:
            to_write= str(ans)
            file.write(to_write)

        if args.perform_evaluation:
            #TODO :better compatibility
            
            #wrapp model and call function
        
            # TODO : save the evaluation parameters (folder)
            device="cuda:0"
            test_model=TorchBatchModelWrapper(args.model_type,args.weight_name,device=device)
            kwargs={
                "device":device,
                "dataset_path":os.path.join(os.getcwd(), "data/cifar-10-batches-py/test_batch"),
                "batch_size":1,
                "num_classes-dataset":10,
                "n_ways":5,
                "n_shots":5,
                "n_runs":1000, 
                "n_queries":15,
                "batch_size_fs":20,
                "classifier_type":"ncm",
                "number_neiboors":5,
                "resolution_input":(input_resolution,input_resolution)
            }
            kwargs=argparse.Namespace(**kwargs)
            mean,std,time=evaluate_model(test_model,kwargs)


        # generate onnx
        path_model=resolution_folder/ f"{model_name}-{input_resolution}_{input_resolution}.onnx"
        path_model_simp=resolution_folder/ f"simp-{model_name}-{input_resolution}_{input_resolution}.onnx"
        
        
        torch.onnx.export(model, dummy_input, path_model, verbose=True, opset_version=10, output_names=[args.output_names])
        
        #load onnx
        onnx_model = onnx.load(path_model)

        # convert model
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        
        onnx.save(model_simp,path_model_simp)



if __name__ == "__main__":
    
    # Define the command line arguments for the script
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-resolution", nargs='+', required=True, help="Input resolution(s) of the images (squared images), ex: 32 64")
    parser.add_argument("--model_source",type=str,default="pytorch-hub")
    parser.add_argument("--model-type",type=str, required=True, help="Specification of the model")
    parser.add_argument("--weight-name", type=str, required=True, help="how to get the weight (in function of the model loaded)")
    parser.add_argument("--weight-description", required=True, help="Description of the weight file")
    parser.add_argument("--output-names",default="Output", help="Name of the output layer")
    #TODO : add evaluation args
    parser.add_argument("--perform-evaluation",action='store_true', help="if specified, will perform inference")
    # Parse the command line arguments
    args = parser.parse_args()
    model_to_onnx(args)


    