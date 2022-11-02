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


from utils import opencv_interface
from possible_models import get_model, load_model_weights, predict_class_moving_avg
from preprocess import image_preprocess, feature_preprocess

print("import done")

# addr_cam = "rtsp://admin:brain2021@10.29.232.40"
# cap = cv2.VideoCapture(addr_cam)


def save_feature(save_to, current_classe, feature_to_save):
    """
    save a given feature of a given class in a dictionnary
        parameters :
            save_to(dict) : contains feature, present class, mean features, and snapshots
            current_class : class of the feature
            feature_to_save : feature to be saved
    """
    if current_classe not in save_to["registered_classes"]:
        save_to["registered_classes"].append(current_classe)
        save_to["shot_list"].append(feature_to_save)
    else:
        save_to["shot_list"][current_classe] = torch.cat(
            (save_to["shot_list"][current_classe], feature_to_save), dim=0
        )
        print("------------:", save_to["shot_list"][current_classe].shape)


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
MODEL_NAME = "knn"
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
    empty_data = {
        "registered_classes": [],
        "shot_frames": [[] for i in range(len(possible_input))],
        "shot_list": [],
        "mean_features": [],
    }
    data = copy.deepcopy(empty_data)

    # CV2 related constant
    cap = cv2.VideoCapture(0)
    cv_interface = opencv_interface(cap, SCALE, RES_OUTPUT, FONT)
    # model related
    model = get_model("resnet12", MODEL_SPECS).to(DEVICE)
    load_model_weights(model, PATH_MODEL, DEVICE)

    while True:
        cv_interface.read_frame()

        new_frame_time = time.time()
        # print('clock: ', clock)
        fps = int(1 / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time

        if clock_m <= clock_init:
            frame = cv_interface.get_image()
            img = image_preprocess(frame).to(DEVICE)
            _, features = model(img.unsqueeze(0))
            data["mean_features"].append(features.detach().to(DEVICE))
            if clock_m == clock_init:
                data["mean_features"] = torch.cat(data["mean_features"], dim=0)
                data["mean_features"] = data["mean_features"].mean(dim=0)

            cv_interface.put_text("Initialization")

            clock_m += 1

        key = cv2.waitKey(33) & 0xFF

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
            img = image_preprocess(frame).to(DEVICE)
            _, features = model(img.unsqueeze(0))

            # preprocess features
            features = feature_preprocess(
                features, mean_base_features=data["mean_features"]
            )
            print("features:", features.shape)
            if key in possible_input:
                print(f"saving snapshot of class {classe}")
                cv_interface.add_snapshot(data, classe)
            # add the representation to the class

            save_feature(data, classe, features)
            if abs(clock - last_detected) < 10:
                do_registration = True
                text = f'Class :{classe} registered. \
                Number of shots: {len(data["shot_frames"][classe])}'
                cv_interface.put_text(text)
            else:
                do_registration = False

        # reset action
        if key == ord("r"):
            do_registration = False
            do_inference = False
            mean_features = data["mean_features"]
            data = copy.deepcopy(empty_data)
            data["mean_features"] = mean_features
            reset_clock = 0
            do_reset = True

        if do_reset:
            cv_interface.put_text("Resnet background inference")
            reset_clock += 1
            if reset_clock > 20:
                do_reset = False

        # inference actionfont
        if key == ord("i") and len(data["shot_list"]) > 0:
            do_inference = True
            probabilities = None

        # perform infernece
        if do_inference and clock_m > clock_init and not do_reset:
            frame = cv_interface.get_image()
            img = image_preprocess(frame).to(DEVICE)

            classe_prediction, probabilities = predict_class_moving_avg(
                img, data, model, MODEL_NAME, probabilities
            )

            print("probabilities after exp moving average:", probabilities)
            cv_interface.put_text(f"Object is from class :{classe_prediction}")
            # f'Probabilities :{list(map(lambda x:np.round(x, 2), probabilities.tolist()))}'
            cv_interface.draw_indicator(probabilities, data["shot_frames"])

        # interface
        cv_interface.put_text(f"fps:{fps}", bottom_pos_x=0.05, bottom_pos_y=0.1)
        cv_interface.put_text(f"clock:{clock}", bottom_pos_x=0.8, bottom_pos_y=0.1)
        cv_interface.show()

        clock += 1
        # reset clock
        # if clock == 100: clock = 0
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv_interface.close()


launch_demo()
