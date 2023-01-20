"""
adapted from : 
EASY - Ensemble Augmented-Shot Y-shaped Learning: State-Of-The-Art Few-Shot Classification with Simple Ingredients.
(https://github.com/ybendou/easy)
(to load a model without training)
"""

import argparse
import os
import sys



parser = argparse.ArgumentParser(description="""
    Launch the demo/ the evaluation of the dataset. 
""", formatter_class=argparse.RawTextHelpFormatter)

### hyperparameters

#parser.add_argument("--feature-maps", type=int, default=64, help="number of feature maps")
#parser.add_argument("--lr", type=float, default="0.1", help="initial learning rate (negative is for Adam, e.g. -0.001)")
#parser.add_argument("--epochs", type=int, default=350, help="total number of epochs")
#parser.add_argument("--milestones", type=str, default="100", help="milestones for lr scheduler, can be int (then milestones every X epochs) or list. 0 means no milestones")
#parser.add_argument("--gamma", type=float, default=-1., help="multiplier for lr at milestones")
#parser.add_argument("--cosine", action="store_true", help="use cosine annealing scheduler with args.milestones as T_max")
##parser.add_argument("--mixup", action="store_true", help="use of mixup since beginning")
#parser.add_argument("--mm", action="store_true", help="to be used in combination with mixup only: use manifold_mixup instead of classical mixup")
#parser.add_argument("--label-smoothing", type=float, default=0, help="use label smoothing with this value")
#parser.add_argument("--dropout", type=float, default=0, help="use dropout")
#parser.add_argument("--rotations", action="store_true", help="use of rotations self-supervision during training")
#parser.add_argument("--model", type=str, default="ResNet18", help="model to train")
#parser.add_argument("--preprocessing", type=str, default="", help="preprocessing sequence for few shot, can contain R:relu P:sqrt E:sphering and M:centering")
#parser.add_argument("--postprocessing", type=str, default="", help="postprocessing sequence for few shot, can contain R:relu P:sqrt E:sphering and M:centering")

#parser.add_argument("--manifold-mixup", type=int, default="0", help="deploy manifold mixup as fine-tuning as in S2M2R for the given number of epochs")
#parser.add_argument("--temperature", type=float, default=1., help="multiplication factor before softmax when using episodic")
#parser.add_argument("--ema", type=float, default=0, help="use exponential moving average with specified decay (default, 0 which means do not use)")

### pytorch options

#parser.add_argument("--deterministic", action="store_true", help="use desterministic randomness for reproducibility")

### run options
#parser.add_argument("--skip-epochs", type=int, default="0", help="number of epochs to skip before evaluating few-shot performance")
#parser.add_argument("--runs", type=int, default=1, help="number of runs")
#parser.add_argument("--quiet", action="store_true", help="prevent too much display of info")
#parser.add_argument("--output", type=str, default="", help="output file to write")
#parser.add_argument("--save-features", type=str, default="", help="save features to file")
#parser.add_argument("--save-model", type=str, default="", help="save model to file")
#parser.add_argument("--test-features", type=str, default="", help="test features and exit")
#parser.add_argument("--load-model", type=str, default="", help="load model from file")
#parser.add_argument("--seed", type=int, default=-1, help="set random seed manually, and also use deterministic approach")
#parser.add_argument("--wandb", type=str, default='', help="Report to wandb, input is the entity name")

#
#parser.add_argument("--ncm-loss", action="store_true", help="use ncm output instead of linear")
#
#parser.add_argument("--episodes-per-epoch", type=int, default=100, help="number of episodes per epoch")
# only for transductive, used with "test-features"
#parser.add_argument("--transductive", action="store_true", help ="test features in transductive setting")
#parser.add_argument("--transductive-softkmeans", action="store_true", help="use softkmeans for few-shot transductive")
#parser.add_argument("--transductive-n-iter", type=int, default=50, help="number of iterations for few-shot transductive")
#parser.add_argument("--transductive-n-iter-sinkhorn", type=int, default=200, help="number of iterations of sinkhorn for few-shot transductive")
#parser.add_argument("--transductive-temperature", type=float, default=14, help="temperature for few-shot transductive")
#parser.add_argument("--transductive-temperature-softkmeans", type=float, default=20, help="temperature for few-shot transductive is using softkmeans")
#parser.add_argument("--transductive-alpha", type=float, default=0.84, help="momentum for few-shot transductive")
#parser.add_argument("--transductive-cosine", action="store_true", help="use cosine similarity for few-shot evaluation")



def parse_dataset_feature(parser):

    parser.add_argument("--dataset-path", type=str, default="/data/", help="dataset path")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--num-classes-dataset", type=int, default=10, help="number of class in dataset")

    
    # to be incoreporate (pytorch dataset): 
    #parser.add_argument("--dataset", type=str, default="cifarfs", help="dataset to use")
    #parser.add_argument("--episodic", action="store_true", help="use episodic training")#legacy code, pytorch training dataset are constructed but not used
    #parser.add_argument("--dataset-size", type=int, default=-1, help="number of training samples (using a subset for classical classification, and reducing size of epochs for few-shot)")
    
def parse_few_shot_eval_params(parser):
    ### few-shot parameters

   
    parser.add_argument("--n-ways", type=int, default=5, help="number of few-shot ways")
    parser.add_argument("--n-shots", type=int, default=5, help="how many shots per few-shot run, can be int or list of ints. In case of episodic training, use first item of list as number of shots.")
    parser.add_argument("--n-runs", type=int, default=1000, help="number of few-shot runs")
    parser.add_argument("--n-queries", type=int, default=15, help="number of few-shot queries")
    parser.add_argument("--batch-size-fs",type=int,default=20)
    #to be incorporate (to evaluation and demonstration):
    #parser.add_argument("--sample-aug", type=int, default=1, help="number of versions of support/query samples (using random crop) 1 means no augmentation")
    

def parse_backbone_params(parser):
   
   
    #usefull only for pytorch
    parser.add_argument("--backbone_type",default="cifar_small",help="model to load")
    
    #only usefull for the pynk
    parser.add_argument("--path_bit",default="/home/xilinx/jupyter_notebooks/l20leche/base_tensil_hdmi.bit",type=str)
    parser.add_argument("--path_tmodel",default="/home/xilinx/resnet12_32_32_small_onnx_pynqz1.tmodel",type=str)
    
    #only usefull for onnx

    parser.add_argument("--path-onnx",default="weight/resnet12_32_32_64.onnx",type=str)


def parse_fs_model_params(parser):
    parser.add_argument("--classifier_type",default="ncm",type=str)
    parser.add_argument("--number_neiboors",default=5,type=int)
    
def parse_mode_args(parser):
    #parser.add_argument("--mode",type=str,default="", help="in what mode should it run (demo or perf)")
    parser.add_argument("--framework_backbone",type=str,default="tensil_model", help="wich module should we use")

def parse_hyperparameter_demonstration(parser):
    parser.add_argument("--camera-specification",type=str,default="0")
    parser.add_argument("--no-display",action="store_true")
    parser.add_argument("--save-video",action="store_true")
    
    parser.add_argument("--video-format",type=str,default="DIVX")
    parser.add_argument("--max_number_of_frame",type=int)
    parser.add_argument("--use-saved-sample",action="store_true")
    parser.add_argument("--path_shots_video",type=str,default="data/catvsdog")

    parser.add_argument("--verbose",action="store_true")
    parser.add_argument("--button-keyboard", default="keyboard")

    
#generl paramters
parse_mode_args(parser)
    
#usefull only for demonstration
parse_hyperparameter_demonstration(parser)

#usefull only for performance evaluation
parse_dataset_feature(parser)
parse_few_shot_eval_params(parser)

#always usefull
parse_backbone_params(parser)
parse_fs_model_params(parser)


parser.add_argument("--device", type=str, default="cuda:0", help="device(s) to use, for multiple GPUs try cuda:ijk, will not work with 10+ GPUs")


try :
    get_ipython()
    args = parser.parse_args(args=[])
except :
    args = parser.parse_args()
    
print("input args : ",args)

### process arguments
if args.camera_specification=="None":
    args.camera_specification=None
else:
    try:
        args.camera_specification=int(args.camera_specification)
    except:
        print("using a video file")

args.dataset_path=os.path.join(os.getcwd(),args.dataset_path)
if args.framework_backbone=="pytorch_batch":
    
    # if args.dataset_device == "":
    #     args.dataset_device = args.device

    #if args.dataset_path[-1] != '/':
    #    args.dataset_path += "/"

    if args.device[:5] == "cuda:" and len(args.device) > 5:
        args.devices = []
        for i in range(len(args.device) - 5):
            args.devices.append(int(args.device[i+5]))
        args.device = args.device[:6]
    else:
        args.devices = [args.device]
        
    #backbone arguments :
    args.backbone_specs={
        "type":args.framework_backbone,
        "device":args.device,
        "model_name":"resnet12",
        "kwargs": {
         "input_shape": [3, 32, 32],
         "num_classes": 64,  # 351,
         "few_shot": True,
         "rotations": False,
     },
    }
    if args.backbone_type=="cifar_small":
        args.backbone_specs["path"]="weight/smallcifar1.pt1"
        args.backbone_specs["kwargs"]["feature_maps"]=45

    elif args.backbone_type=="cifar":
        args.backbone_specs["path"]="weight/cifar1.pt1"
        args.backbone_specs["kwargs"]["feature_maps"]=64

    elif args.backbone_type=="cifar_tiny":
        args.backbone_specs["path"]="weight/tinycifar1.pt1"
        args.backbone_specs["kwargs"]["feature_maps"]=32
    else:
        raise UserWarning("parameters for this backbone type is not completed in args.py")
    print(args.backbone_specs)
    

elif args.framework_backbone=="tensil_model":
    #backbone arguments :
    args.backbone_specs={
        "type":args.framework_backbone,
        "path_bit":args.path_bit,
        "path_tmodel":args.path_tmodel
    }
elif args.framework_backbone=="onnx":
    args.backbone_specs={
        "type":args.framework_backbone,
        "path_onnx":args.path_onnx
    }



if args.device=="pynk":
    print("adding path to local variable")
    sys.path.append('/home/xilinx')
    sys.path.append('/home/xilinx/jupyter_notebooks/l20leche')
    sys.path.append('/usr/local/lib/python3.8/dist-packages')
    sys.path.append('/root/.ipython')
    sys.path.append('/usr/local/share/pynq-venv/lib/python3.8/site-packages/IPython/extensions')
    sys.path.append('/usr/lib/python3/dist-packages')
    sys.path.append('/usr/local/share/pynq-venv/lib/python3.8/site-packages')
    sys.path.append('/usr/lib/python3.8/dist-packages')



#classifier arguments
args.classifier_specs={"model_name":args.classifier_type}

if args.classifier_type=="knn":
    args.classifier_specs["kwargs"]={"number_neighboors":args.number_neiboors}
    
    
# if args.seed == -1:
#     args.seed = random.randint(0, 1000000000)


# try:
#     milestone = int(args.milestones)
#     args.milestones = list(np.arange(milestone, args.epochs + args.manifold_mixup, milestone))
# except:
#     args.milestones = eval(args.milestones)
# if args.milestones == [] and args.cosine:
#     args.milestones = [args.epochs + args.manifold_mixup]

# if args.gamma == -1:
#     if args.cosine:
#         args.gamma = 1.
#     else:
#         args.gamma = 0.1

# if args.mm:
#     args.mixup = True
    
# print("args, ", end='')