"""
script will load models, generate torchinfos and onnx (simplified version) for each resolution

python model_to_onnx.py --input-resolution 32 64 84 --save-name "tiny_miniimagenet" --model-type "easy_resnet12_tiny" --model-specification "weights/tinymini1.pt1" --weight-description "weight from easy repo"
python model_to_onnx.py --input-resolution 32 64 84 --save-name "small_miniimagenet" --model-type "easy_resnet12_small" --model-specification "weights/smallmini1.pt1" --weight-description "weight from easy repo"
python model_to_onnx.py --input-resolution 32 --save-name "small_cifar" --model-type "easy_resnet12_small" --model-specification "weights/smallcifar1.pt1" --weight-description "weight from easy repo"
python model_to_onnx.py --input-resolution 32 --save-name "tiny_cifar" --model-type "easy_resnet12_tiny" --model-specification "weights/tinycifar1.pt1" --weight-description "weight from easy repo"



# exemple of command line loading network from pytorch hub (also convert the classification head, you should adapt the script to delete thoses nodes)
python model_to_onnx.py --input-resolution 32 64 84 128  --model-type "mobilenet_v2" --model-specification "pretrained" --weight-description "weight trained on imagenet"  --from-hub
python model_to_onnx.py --input-resolution 32 64 84 128  --model-type "mnasnet0_5" --model-specification "random_init" --weight-description "weight random"  --from-hub

not implemented in Tensil
python model_to_onnx.py --input-resolution 32 64 128  --model-type "nvidia_efficientnet_b0" --model-specification "random_init" --weight-description "weight random"  --from-hub
python model_to_onnx.py --input-resolution 32 64 128  --model-type "shufflenet_v2_x0_5" --model-specification "pretrained" --weight-description "weight trained on imagenet"  --from-hub
python model_to_onnx.py --input-resolution 32 64 128  --model-type "squeezenet1_1" --model-specification "pretrained" --weight-description "weight trained on imagenet"  --from-hub
python model_to_onnx.py --input-resolution 32 64 128  --model-type "inception_v3" --model-specification "pretrained" --weight-description "weight trained on imagenet"  --from-hub
python model_to_onnx.py --input-resolution 32 64 128  --model-type "googlenet" --model-specification "pretrained" --weight-description "weight trained on imagenet"  --from-hub
python model_to_onnx.py --input-resolution 32 64  --model-type "densenet121" --model-specification "pretrained" --weight-description "weight trained on imagenet"  --from-hub
python model_to_onnx.py --input-resolution 32 64 128  --model-type "nvidia_gpunet" --model-specification "pretrained" --weight-description "weight trained on imagenet"  --from-hub


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
import warnings

from backbone_loader.backbone_pytorch.model import get_model
from backbone_loader.backbone_loader_pytorch import TorchBatchModelWrapper


def save_weight_description(path_description, model_specification, weight_desc):
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


def replace_reduce_mean(
    onnx_model,
):
    """
    Replace all reduce_mean operation with GlobalAveragePool if they act on the last dimentions of the tensor
    do not change the name of this operation.
    Suppose that attribute has the following element :
        name: "axes"
        ints: -2 /2
        ints: -1 /3
        type: INTS

    Possible amelioration :
        - use more onnx helper
        - fix case where will cause another reshape to be
        - fix the case where the feature size is not the same as the last one in the reduce mean
    """

    if onnx_model.ir_version != 5:
        warnings.warn(
            "conversion of onnx was tested with ir_version 5, may not work with another one"
        )
    if len(onnx_model.graph.output) != 1:
        raise ValueError("only one output is supported")

    # find the batch size and output size
    output = onnx_model.graph.output[0]
    print(output.type)
    shape_output = output.type.tensor_type.shape.dim

    if len(shape_output) != 2:
        raise ValueError("only support output of shape (batch_size, output_size)")

    batch_size, num_feature_output = (
        shape_output[0].dim_value,
        shape_output[1].dim_value,
    )
    print(batch_size)

    for pos, node in enumerate(onnx_model.graph.node):
        if node.name.find("ReduceMean") < 0:
            continue

        print("attributes of node :")
        print(node.attribute)

        # check if node operate on the last op
        do_replace_mean = False
        number_attribute = len(node.attribute)
        index_keep_dims = -1
        for i in range(number_attribute):
            attribute = node.attribute[i]

            if attribute.name == "axes":
                x, y = attribute.ints
                if (x == 2 and y == 3) or (x == 3 and y == 2):
                    do_replace_mean = True
                if (x == -2 and y == -1) or (x == -1 and y == -2):
                    do_replace_mean = True
            if attribute.name == "keepdims":
                index_keep_dims = i

        # replace the node if needed
        if do_replace_mean:
            print("Replacing ReduceMean operation with GlobalAveragePool")
            if index_keep_dims >= 0:
                if node.attribute[index_keep_dims].i == 0:
                    print("adding one reshape layer ")
                    old_output_name = node.output.pop()

                    # reshape dimentions
                    reshape_data = onnx.helper.make_tensor(
                        name="Reshape_dim",
                        data_type=onnx.TensorProto.INT64,
                        dims=[2],
                        vals=np.array([batch_size, num_feature_output])
                        .astype(np.int64)
                        .tobytes(),
                        raw=True,
                    )
                    onnx_model.graph.initializer.append(reshape_data)

                    type_output = onnx.helper.make_tensor_type_proto(
                        onnx.TensorProto.FLOAT, shape=[batch_size, num_feature_output]
                    )

                    # print(type_output)

                    # new output layer
                    new_output_name = "reshape_output"  # onnx.helper.make_tensor_value_info("reshape_input", type_output)

                    node.output.append(new_output_name)
                    new_node = onnx.helper.make_node(
                        op_type="Reshape",
                        name=f"custom_resahpe_{pos}",
                        inputs=[new_output_name, "Reshape_dim"],
                        outputs=[old_output_name],
                    )

                    onnx_model.graph.node.insert(pos + 1, new_node)
            else:
                print("keep dim was not found")
                assert False
            for i in range(number_attribute):
                node.attribute.pop()

            node.op_type = "GlobalAveragePool"
            node.name = (
                "GlobalAveragePool" + node.name[len("ReduceMean") :]
            )  # ReduceMean_32 -> GlobalAveragePool_32
        else:
            print(
                "cant replace ReduceMean (do not recognize that the dimention are last)"
            )
    return onnx_model


def model_to_onnx(args):
    # create model path
    # one model = sevral possible resolutions
    # generate summary and save it

    # args arguments
    input_resolution = [int(i) for i in args.input_resolution]

    if args.from_hub:
        model_name = f"hub_{args.model_type}_{args.model_specification}"
    else:
        model_name = f"{args.model_type}"

    # handling path
    model = get_model(args.model_type, args.model_specification)

    parent_path = Path.cwd() / "onnx"
    parent_path.mkdir(parents=False, exist_ok=True)
    info_path = parent_path / "infos"
    info_path.mkdir(parents=False, exist_ok=True)
    weight_desc_file = info_path / f"{model_name}_description.json"

    # Saving
    save_weight_description(
        weight_desc_file, args.model_specification, args.weight_description
    )

    min_res = 32
    max_res = 100  # max(input_resolution)#TODO res is to be set in function of maximum identified resolution for fpga

    number_eval = 100
    step_res = 10
    possible_res = [i for i in range(min_res, max_res, step_res)]
    time_execution = np.zeros((len(possible_res), number_eval))

    # first emtpy call (in one occasion, i've observed that it was slower)
    dummy_input = torch.randn(1, 3, min_res, min_res, device="cpu")
    _ = model(dummy_input)

    if args.check_perf:
        for i_res, res in tqdm(enumerate(possible_res), total=len(possible_res)):
            print("res : ", res)
            dummy_input = torch.randn(1, 3, res, res, device="cpu")

            for i in range(number_eval):
                t = time.time()
                _ = model(dummy_input)
                time_execution[i_res, i] = time.time() - t

        np.savez(
            info_path / f"exec_time_{model_name}.npy",
            possible_res=possible_res,
            time_execution=time_execution,
        )

        print("possible resolutions : ", possible_res)
        print("mean execution time : ", np.mean(time_execution, axis=-1))
        print("mean fps : ", 1 / np.mean(time_execution, axis=-1))

    # for each input, create the corresponding file

    for input_resolution in input_resolution:
        print("exporting res : ", input_resolution)
        resolution_folder = parent_path / f"{input_resolution}x{input_resolution}"
        resolution_folder.mkdir(parents=False, exist_ok=True)

        ans = torchinfo.summary(
            model,
            (3, input_resolution, input_resolution),
            batch_dim=0,
            verbose=0,
            device="cpu",
            col_names=[
                "input_size",
                "output_size",
                "num_params",
                "kernel_size",
                "mult_adds",
            ],
        )

        dummy_input = torch.randn(
            1, 3, input_resolution, input_resolution, device="cpu"
        )

        with open(
            info_path
            / f"{args.save_name}_{input_resolution}x{input_resolution}_torchinfo.txt",
            "w",
            encoding="utf-8",
        ) as file:
            to_write = str(ans)
            file.write(to_write)

        # generate onnx
        path_model = (
            resolution_folder
            / f"{args.save_name}_{input_resolution}x{input_resolution}.onnx"
        )  # f"{model_name}_{weight_name}_{input_resolution}_{input_resolution}.onnx"
        # path_model_simp=resolution_folder/ f"simp_{model_name}_{input_resolution}_{input_resolution}.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            path_model,
            verbose=False,
            opset_version=10,
            output_names=[args.output_names],
        )

        # load onnx

        onnx_model = onnx.load(path_model)
        onnx_model = replace_reduce_mean(onnx_model)
        # convert model
        model_simp, check = simplify(onnx_model)
        print("model was simplified")
        assert check, "Simplified ONNX model could not be validated"
        # path_model_simp=resolution_folder/ f"simp-{model_name}-{input_resolution}_{input_resolution}.onnx"#if one wants to test difference
        onnx.save(model_simp, path_model)


if __name__ == "__main__":
    # Define the command line arguments for the script

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-resolution",
        nargs="+",
        required=True,
        help="Input resolution(s) of the images (squared images), ex: 32 64",
    )
    parser.add_argument(
        "--from-hub", action="store_true", help="if true, add hub- to name"
    )
    parser.add_argument(
        "--model-type", type=str, required=True, help="Specification of the model"
    )
    parser.add_argument(
        "--model-specification",
        type=str,
        required=True,
        help="additional specs for model. 1. easy_resnet, path to weight 2. one of hardcoded kwargs in model.py",
    )
    parser.add_argument(
        "--weight-description", required=True, help="Description of the weight file"
    )
    parser.add_argument(
        "--output-names", default="Output", help="Name of the output layer"
    )
    parser.add_argument(
        "--check-perf",
        action="store_true",
        help="if specified, will perform inference evaluations",
    )
    parser.add_argument("--save-name", required=True, help="Name of the save model")
    # parser.add_argument("--perform-evaluation",action='store_true', help="if specified, will perform inference")
    # Parse the command line arguments
    args = parser.parse_args()
    model_to_onnx(args)
