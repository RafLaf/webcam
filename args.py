"""
adapted from : 
EASY - Ensemble Augmented-Shot Y-shaped Learning: State-Of-The-Art Few-Shot Classification with Simple Ingredients.
(https://github.com/ybendou/easy)
(to load a model without training)
"""

import argparse
import os
import sys


def parse_evaluation_args(parser):

    eval_group = parser.add_argument_group(
        "evaluation", description="evaluation specific arguments"
    )
    # dataset features
    eval_group.add_argument(
        "--dataset-path", type=str, default="data/", help="dataset path"
    )
    eval_group.add_argument("--batch-size", type=int, default=1, help="batch size")
    eval_group.add_argument(
        "--num-classes", type=int, default=10, help="number of class in dataset"
    )

    eval_group.add_argument(
        "--sample-per-class",
        type=int,
        default=1000,
        help=" number of sample to take into acount",
    )

    ### few-shot parameters
    eval_group.add_argument(
        "--n-ways", type=int, default=5, help="number of few-shot ways"
    )
    eval_group.add_argument(
        "--n-shots",
        type=int,
        default=5,
        help="how many shots per few-shot run, can be int or list of ints. In case of episodic training, use first item of list as number of shots.",
    )
    eval_group.add_argument(
        "--n-runs", type=int, default=1000, help="number of few-shot runs"
    )
    eval_group.add_argument(
        "--n-queries", type=int, default=15, help="number of few-shot queries"
    )
    eval_group.add_argument("--batch-size-fs", type=int, default=20)
    # to be incorporate (to evaluation and demonstration):
    # parser.add_argument("--sample-aug", type=int, default=1, help="number of versions of support/query samples (using random crop) 1 means no augmentation")


def parse_model_params(parser):

    model_args = parser.add_argument_group(
        "model", description="model specific arguments"
    )
    # classification head
    model_args.add_argument("--resolution-input", default=32)
    model_args.add_argument("--classifier_type", default="ncm", type=str)
    model_args.add_argument("--number_neiboors", default=5, type=int)

    # usefull only for pytorch

    framework_submodules = parser.add_subparsers(
        help="option relative to framwork, including how the backbone is loaded",
        dest="framework_backbone",
    )

    pytorch_parser = framework_submodules.add_parser(
        "pytorch", help="pytorch specific arguments"
    )

    pytorch_parser.add_argument(
        "--device-pytorch",
        type=str,
        default="cuda:0",
        help="for pytorch only. Device on wich the backbone will be run",
    )

    pytorch_parser.add_argument("--path-pytorch-weight", default=None, type=str)
    pytorch_parser.add_argument(
        "--backbone-type",
        default="easy_resnet12",
        help="model to load. available for easy : easy_resnet12/easy_resnet12_tiny/easy_resnet12_small",
    )

    # only usefull for the pynk
    pynq_parser = framework_submodules.add_parser(
        "tensil", help="pynq specific arguments"
    )

    pynq_parser.add_argument(
        "--path_bit",
        default="/home/xilinx/jupyter_notebooks/l20leche/base_tensil_hdmi.bit",
        type=str,
    )
    pynq_parser.add_argument(
        "--path_tarch",
        default="/home/xilinx/jupyter_notebooks/l20leche/resnet12_32_32_small_onnx_pynqz1.tarch",
        type=str,
    )
    pynq_parser.add_argument(
        "--path_tmodel",
        default="/home/xilinx/resnet12_32_32_small_onnx_pynqz1.tmodel",
        type=str,
    )

    # only usefull for onnx
    onnx_parser = framework_submodules.add_parser(
        "onnx", help="onnx specific arguments"
    )

    onnx_parser.add_argument(
        "--path-onnx", default="weight/resnet12_32_32_64.onnx", type=str
    )
    # only usefull for pytorch
    onnx_parser.add_argument("--path-pytorch-weight", default=None, type=str)


def parse_args_demonstration(parser):
    demonstration_arguments = parser.add_argument_group(
        "input / output", description="input output specific arguments"
    )
    demonstration_arguments.add_argument(
        "--camera-specification", type=str, default="0"
    )
    demonstration_arguments.add_argument("--no-display", action="store_true")
    demonstration_arguments.add_argument("--save-video", action="store_true")
    demonstration_arguments.add_argument("--hdmi-display", action="store_true")
    demonstration_arguments.add_argument("--video-format", type=str, default="DIVX")
    demonstration_arguments.add_argument("--max_number_of_frame", type=int)
    demonstration_arguments.add_argument("--use-saved-sample", action="store_true")
    demonstration_arguments.add_argument(
        "--path_shots_video", type=str, default="data/catvsdog"
    )
    demonstration_arguments.add_argument("--verbose", action="store_true")
    demonstration_arguments.add_argument("--button-keyboard", default="keyboard")


def process_arguments(args):
    """
    process relative to both demo and cifar evaluation
    """

    if args.framework_backbone == "pytorch":

        # backbone arguments :
        args.backbone_specs = {
            "type": args.framework_backbone,
            "device": args.device_pytorch,
            "model_name": args.backbone_type,
        }

        # weights hardcoded path convinience

        if args.path_pytorch_weight is None:
            print("no weight provided, using hardcoded path")

            if args.backbone_type == "easy_resnet12_small":
                args.backbone_specs["weight"] = "weight/smallcifar1.pt1"
            elif args.backbone_type == "easy_resnet12":
                args.backbone_specs["weight"] = "weight/cifar1.pt1"
            elif args.backbone_type == "easy-resnet12-tiny":
                args.backbone_specs["weight"] = "weight/tinycifar1.pt1"
            else:
                raise UserWarning(
                    f"weights for {args.backbone_type} is not hardcoded, provide the path yourself or check name validity"
                )
        else:
            args.backbone_specs["weight"] = args.path_pytorch_weight
        print(args.backbone_specs)

    elif args.framework_backbone == "tensil":
        # backbone arguments :
        from pynq import Overlay

        args.overlay = Overlay(args.path_bit)
        args.backbone_specs = {
            "type": args.framework_backbone,
            "overlay": args.overlay,
            "tmodel": args.path_tmodel,
            "path_bit": args.path_bit,
            "path_tmodel": args.path_tmodel,

        }

        # # TODO : delete unused path
        # print("adding path to local variable")
        # sys.path.append("/home/xilinx")
        # sys.path.append("/home/xilinx/jupyter_notebooks/l20leche")
        # sys.path.append("/usr/local/lib/python3.8/dist-packages")
        # sys.path.append("/root/.ipython")
        # sys.path.append(
        #     "/usr/local/share/pynq-venv/lib/python3.8/site-packages/IPython/extensions"
        # )
        # sys.path.append("/usr/lib/python3/dist-packages")
        # sys.path.append("/usr/local/share/pynq-venv/lib/python3.8/site-packages")
        # sys.path.append("/usr/lib/python3.8/dist-packages")

    elif args.framework_backbone == "onnx":
        args.backbone_specs = {
            "type": args.framework_backbone,
            "path_onnx": args.path_onnx,
        }

        # classifier arguments
        args.classifier_specs = {"model_name": args.classifier_type}

        if args.classifier_type == "knn":
            args.classifier_specs["kwargs"] = {
                "number_neighboors": args.number_neiboors
            }


def process_args_evaluation(args):
    process_arguments(args)
    args.dataset_path = os.path.join(os.getcwd(), args.dataset_path)
    return args


def get_args_evaluation():
    parser = argparse.ArgumentParser(
        description="""
        Launch the evaluation of the dataset
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # specifics args
    parse_evaluation_args(parser)
    parse_model_params(parser)

    args = parser.parse_args()
    process_args_evaluation(args)

    print("input args : ", args)
    return args


def process_args_demo(args):
    process_arguments(args)
    ### process remaining arguments

    # resolution
    if type(args.resolution_input) is int:
        args.resolution_input = (args.resolution_input, args.resolution_input)
    elif len(args.resolution_input) == 1:
        res_x = args.resolution_input[0]
        args.resolution_input = (res_x, res_x)
    args.resolution_input = tuple(args.resolution_input)

    if args.camera_specification == "None":
        args.camera_specification = None
    else:
        try:
            args.camera_specification = int(args.camera_specification)
        except:
            print("using a video file")

    print("input args : ", args)
    return args


def get_args_demo():

    parser = argparse.ArgumentParser(
        description="""
        Launch the evaluation of the dataset
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # specifics args
    parse_args_demonstration(parser)
    parse_model_params(parser)

    args = parser.parse_args()
    print("input args : ", args)
    args = process_args_demo(args)
    return args


# print(args.dataset_path)
