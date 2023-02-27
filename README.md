# Low-latency Few-shot classification 

This repository contains the code to perform online Few shot with a webcam on fpga.


## demo : 

    Press 0 1 or 2 to associate current camera feed to a shot.

    You can add multiple shots by pressing the same class (0, 1 or 2...)
    Press 'i' to start infering.

    To run the code :
        python main.py + args

    exemples :
    - run demo with onnx and camera (note that onnx should be downloaded and correspond to the specified resolution)
        python main.py --framework_backbone onnx --camera-specification 0 --no-display --save-video --resolution-input 32 --path-onnx weight/resnet12_32_32_64.onnx
    - run demo with onnx and video (note that onnx should be downloaded and correspond to the specified resolution)
        python main.py --framework_backbone onnx --camera-specification catvsdog.mp4 --no-display --save-video --use-saved-sample --resolution-input 32 --path-onnx weight/resnet12_32_32_64.onnx
    - run demo with opencv output and pytorch on cpu (note that the weight be downloaded and correspond to the specified model)
        python main.py --framework_backbone pytorch --device-pytorch cpu --backbone_type cifar_small --path-pytorch-weight --path-pytorch-weight weight/cifartiny1.pt --resolution-input 64
    - run demo with tensil for 300 frames(path of driver is hardcoded)
        python main.py --framework_backbone tensil --camera-specification catvsdog.mp4 --no-display --save-video --use-saved-sample --max_number_of_frame 300 --resolution-input 32 --path-onnx weight/resnet12_32_32_64.onnx
    - run evaluation with onnx 
        python few_shot_evaluation.py --framework_backbone onnx --path-onnx weight/resnet12_32_32_32.onnx --dataset-path data/cifar-10-batches-py/test_batch

        python few_shot_evaluation.py --framework_backbone onnx --path-onnx onnx/32x32/resnet12-tiny-mini1_32x32.onnx --dataset-path data/cifar-10-batches-py/test_batch

![plot](./static/demo_webcam.png)

# Data installation :
## weights of the neural network : 
- see this repo : https://github.com/ybendou/easy

## data :
- used : test set of cifar-10 
## video : 
in order to setup a video
1. download a video and put it in this repo. 
2. put reference images inside a folder with the folowing structure :
    -folder
        -class1_name
        -class2_name
3. add the path as argument when you call the function


# args : 
## specify few_shot model used
    --classifier_type ncm(default)/knn : type of classifier for the feature
    --number_neiboors (default 5) : number of neighboors if the knn algorithm is choosed
    --framework_backbone tensil_model(default)/pytorch/onnx : wich framework should be used
    --resolution-input : resolution of the input data
### tensil specific arguments
    --path_bit : path for the bitstream
    --path_tmodel : path for the tmodel
### pytorch specific argument :
    --backbone_type resnet-usuall/cifar_small(default)/cifar_tiny : used to specify the model used
    --path-pytorch-weight : path of the weight for pytorch
    --device-pytorch pynk/cpu/cuda:0(default)!  device on wich the backbone is stored
### onnx specific argument :
    --path-onnx : path on wich we have the model

## task-related
### inference on cifar-test specifics
    dataset parameters: 
    --dataset-path : path of the data
    -- batch-size (default 1) : batch size for evaluation (only supported with pytorch model)
    -- num-classes-dataset(default 10): number of class in the dataset
    
    model parameters : 
    --nways (default 5) : number of few-shot ways
    --nshots (default 5) : number of shots
    --nruns (default 1000) : number of few-shot runs
    --n-queries (default 15): number of few-shot queries
    --batch-size-fs (default 20): batch_size for the evaluation (on cpu)

### demonstration related
    --camera-specification 0(default)/None : wich camera should be used (can be a path to a file)
    --no-display: don't show output if specified, and output probability in command line instead
    --save_video : if specified, save the video of the demonstration (will have issue if fps are not constant)
    --video_format DIVX(default): see https://docs.opencv.org/3.4/dd/d43/tutorial_py_video_display.html for possible option
    --max_number_of_frame : number of frame after wich to stop the video (for testing purpuses)
    --use-saved-sample: if true, will add samples from a directory once the inference is done
    --path_shots_video : path to the directory containing directories with the shots to be use (inside the directory, one directory per class)

# notes
    - do not put / before path if using relative path