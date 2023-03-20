# Low-latency Few-shot classification 

This repository contains the code to perform online Few shot with a webcam on fpga.


## demo : 

    Press button 0 to take a shot with current class
    Press button 1 to take a shot with next class
    Press button 2 to begin inference
    Press button 3 to reset

    You can add multiple shots by pressing the same class (0, 1 or 2...)
    Press 'i' to start infering.

![plot](./static/demo_webcam.png)

# Data installation :
## weights of the neural network : 
- see this repo for pytorch weights: 
    https://github.com/ybendou/easy

# launch the demonstration on the pynq
Once the bitstream and tensil output has been transfered to the pynq, it's time to launch the domonstration. This demo requires a python environement with numpy, opencv, and the pynq library installed. In order to have the 
right to use the fpga, you need to authentify as root :
```Bash

sudo -i 
cd home/xilinx/few_shot_evaluation.py
main.py --help
performance_evaluation --help
main.py onnx --help
main.py pytorch --help
main.py tensil --help

```

exemples : 
launch the demonstration on pynq :

```
bash
python3 main.py --button-keyboard button --hdmi-display tensil --path_tmodel /home/xilinx/resnet12_32_32_32_onnx_custom_perf.tmodel --path_bit /home/xilinx/jupyter_notebooks/l20leche/Test_Bitstream/1/test.bit
```
    
launch cifar10 evaluation on pynq (only 32x32 networks) :
```
python3 few_shot_evaluation.py --dataset-path /home/xilinx/cifar-10-batches-py/test_batch  tensil --path_tmodel /home/xilinx/resnet12_32_32_32_onnx_custom_perf.tmodel --path_bit /home/xilinx/jupyter_notebooks/l20leche/Test_Bitstream/1/test.bit
```
    



# other setup :

## test the performance of your network :
tensil perform rounding to fixed point16. This may impact a litle bit the accuracy of your model. In order to evaluate this loss, we advice you to compare the results of the evaluation on the dataset cifar-10 before and after quantization :
step to setup the dataset : 
- download the cifar-dataset (you only need the test set) in your cpu (using torchvision function for exemple)
- you should see a binary (cifar-test) under the cifar-10-batches-py folder. This correspond to a binary containing all the cifar-10 image
corresponding to the  testing set

## test of the performance of the demonstration

You may have problem setting up the hdmi output, and want to verify that the demonstration is running well. In order to do that, setup a video simulation of the demo :

1. download a video and put it in this repo. 
2. put reference images inside a folder with the folowing structure :
    -folder
        -class1_name
        -class2_name
3. add the path as argument when you call the function

## conversion of models to onnx : 

basic setup fo onnx exportation is to export it using torch library, and delete all useless nodes with onnx-simplifier. We included a script model_to_onnx.py in order to convert all the pytorch networks implemented in this repo


# args : 
    - for converting model to onnx, exemples are in the docstring of the folder model_to_onnx.py
# Possible pitfall : 
    - in the pynq, always launch the scripts while beeing authentify as root
    - somethimes pynq need to be reset between utilisation in order to use the hdmi
    - path for the pynq specific arguments are often absolute
    - do not put / before path if using relative path
    - should be at least enough elements to form queries + n_shots for the evaluation
