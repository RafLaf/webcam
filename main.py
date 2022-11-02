"""
DEMO of few shot learning:
    connect to the camera and prints prediction in an interface
    press :
    1, 2, 3... : the program will register the current image as an instance of the given class
    i : will start inference
    q : quit the program
"""

import time
import copy
import cv2
import torch# import numpy as np


from graphical_interface import OpencvInterface
from possible_models import get_model,get_features, load_model_weights, \
    predict_class_moving_avg,get_preprocessed_feature

print("import done")

# addr_cam = "rtsp://admin:brain2021@10.29.232.40"
# cap = cv2.VideoCapture(addr_cam)

class DataFewShot:
    """represent the data saved for few shot learning"""

    def __init__(self,num_class):
        self.num_class=num_class
        self.shot_list=list(range(num_class))
        self.registered_classes=[]
        self.mean_repr=[]

    def add_repr(self,classe,repr):
        """
        add the given repr to the given classe
        """
        if classe not in self.registered_classes:
            self.registered_classes.append(classe)
            self.shot_list[classe]=repr
        else:
            #TODO : change dtype to numpy array
            self.shot_list[classe] = torch.cat(
                (self.shot_list[classe], repr), dim=0
            )
            print("------------:", self.shot_list[classe].shape)
    
    def add_mean_repr(self,features,average_feature):
        """
        add a given image to the mean repr of the datas
        """
        
        self.mean_features.append(features.detach().to(DEVICE))
        if average_feature:
            #TODO : change dtype to numpy array
            self.mean_features = torch.cat(self.mean_features, dim=0)
            self.mean_features = self.mean_features.mean(dim=0)
    
    def reset(self):
        """
        reset the saved image, but not the mean repr
        """
        self.shot_list=list(range(self.num_class))
        self.registered_classes=[]



def compute_feature_and_save(frame,backbone,data, current_classe):
    """
    predict and save the given feature of a given class in a dictionnary
        parameters :
            frame(np.ndarray) : current image to predict
            backbone(nn.Module)  backbone to use
            data(DataFewShot) : contains feature, present class, mean features, and snapshots
            current_class : class of the feature
    """
    

    # get features
    features =get_preprocessed_feature(frame,backbone,data.mean_features,DEVICE)
    print("features:", features.shape)

    data.add_repr(current_classe,)


# constant of the program
SCALE = 1
RES_OUTPUT = (1920, 1080)  # resolution = (1280,720)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# model constant
MODEL_SPECS = {
    "feature_maps": 64,
    "input_shape": [3, 84, 84],
    "num_classes": 351,  # 64
    "few_shot": True,
    "rotations": False,
}
PATH_MODEL = "weight/tieredlong1.pt1"

# model parameters
CLASSIFIER_SPECS = {
    "model_name":"knn",
    "args":{
        "number_neighboors":5
    }
}
DEVICE = "cuda:0"


def launch_demo():
    """
    initialize the variable and launch the demo
    """

    # program related constant
    do_inference = False
    do_registration = False
    do_reset = False
    prev_frame_time = time.time()

    possible_input = list(range(48, 53))

    # time related variables
    clock = 0
    clock_m = 0
    clock_init = 20

    # data holding variables

    current_data= DataFewShot(len(possible_input))

    # CV2 related constant
    cap = cv2.VideoCapture(0)
    cv_interface = OpencvInterface(cap, SCALE, RES_OUTPUT, FONT,len(possible_input))
    
    # model related
    model = get_model("resnet12", MODEL_SPECS).to(DEVICE)
    load_model_weights(model, PATH_MODEL, device=DEVICE)

    while True:
        cv_interface.read_frame()

        new_frame_time = time.time()
        # print('clock: ', clock)
        fps = int(1 / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time

        if clock_m <= clock_init:
            frame = cv_interface.get_image()
            features=get_features(frame,model,DEVICE)
            
            current_data.add_mean_repr(features.detach().to(DEVICE),clock_m == clock_init)

            cv_interface.put_text("Initialization")
            clock_m += 1

        key = cv_interface.get_key()

        # shot acquisition
        if (
            (key in possible_input or do_registration)
            and clock_m > clock_init
            and not do_reset
        ):
            do_inference = False
        
            if key in possible_input:
                classe = possible_input.index(key)
                last_detected = clock * 1  # time.time()

            print("class :", classe)
            frame = cv_interface.get_image()
            
            if key in possible_input:
                print(f"saving snapshot of class {classe}")
                cv_interface.add_snapshot(classe)
            
            # add the representation to the class
            features =get_preprocessed_feature(frame,model,current_data.mean_features,DEVICE)
            
            print("features shape:", features.shape)
            
            current_data.add_repr(classe,features)
            compute_feature_and_save(frame,model,current_data, classe)

            if abs(clock - last_detected) < 10:
                do_registration = True
                text = f'Class :{classe} registered. \
                Number of shots: {cv_interface.get_number_snapshot(classe)}'
                cv_interface.put_text(text)
            else:
                do_registration = False

        # reset action
        if key == ord("r"):
            do_registration = False
            do_inference = False
            current_data.reset()
            cv_interface.reset_snapshot()
            reset_clock = 0
            do_reset = True

        if do_reset:
            cv_interface.put_text("Resnet background inference")
            reset_clock += 1
            if reset_clock > 20:
                do_reset = False

        # inference actionfont
        if key == ord("i") and len(current_data.shot_list) > 0:
            do_inference = True
            probabilities = None

        # perform inference
        if do_inference and clock_m > clock_init and not do_reset:
            frame = cv_interface.get_image()
            
            classe_prediction, probabilities = predict_class_moving_avg(
                frame, current_data.shot_list,current_data.mean_features, model, 
                CLASSIFIER_SPECS, probabilities,DEVICE
            )

            print("probabilities after exp moving average:", probabilities)
            cv_interface.put_text(f"Object is from class :{classe_prediction}")
            # f'Probabilities :{list(map(lambda x:np.round(x, 2), probabilities.tolist()))}'
            cv_interface.draw_indicator(probabilities)

        # interface
        cv_interface.put_text(f"fps:{fps}", bottom_pos_x=0.05, bottom_pos_y=0.1)
        cv_interface.put_text(f"clock:{clock}", bottom_pos_x=0.8, bottom_pos_y=0.1)
        cv_interface.show()

        clock += 1
        
        if key == ord("q"):
            break
    cv_interface.close()


launch_demo()
