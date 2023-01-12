# Low-latency Few-shot classification 

This repository contains the code to perform online Few shot with a webcam on fpga.


## demo : 

    Press 0 1 or 2 to associate current camera feed to a shot.

    You can add multiple shots by pressing the same class (0, 1 or 2...)
    Press 'i' to start infering.

    To run the code :
        python main.py + args
    ![plot](./static/demo_webcam.png)
# args : 
## specify few_shot model used
    args :
        --classifier_type ncm(default)/knn : type of classifier for the feature
        --number_neiboors (default 5) : number of neighboors if the knn algorithm is choosed
        --framework_backbone tensil_model(default)/pytorch_batch/onnx : wich framework should be used
        --device pynk/cpu/cuda:0(default)!  device on wich the backbone is stored
### tensil specific arguments
    --path_bit : path for the bitstream
    --path_tmodel : path for the tmodel
### pytorch specific argument :
    --backbone_type cifar/cifar_small(default)/cifar_tiny : used to specify the model used
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