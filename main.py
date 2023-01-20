"""
DEMO of few shot learning:
    connect to the camera and prints prediction in an interface
    press :
    1, 2, 3... : the program will register the current image as an instance of the given class
    i : will start inference
    q : quit the program
"""


#'/usr/local/share/pynq-venv/lib/python3.8/site-packages', '', '', '/usr/lib/python3.8/dist-packages', '', '', '/home/xilinx'
import time
import cv2
import numpy as np
import os
import time

# import cProfile

from graphical_manipulation.graphical_interface import OpencvInterface
from few_shot_model.few_shot_model import FewShotModel
from backbone_loader.backbone_loader import get_model
from few_shot_model.data_few_shot import DataFewShot
from args import args

print("import done")


def compute_and_add_feature_saved_image(
    backbone, cv_interface, current_data, path_sample
):
    classe_idx = 0
    for class_name in os.listdir(path_sample):
        path_class = os.path.join(path_sample, class_name)

        for name_image in os.listdir(path_class):

            path_image = os.path.join(path_class, name_image)
            image = cv2.imread(path_image)
            cv_interface.add_snapshot(classe_idx, frame_to_add=image)
            image = preprocess(image, shape_input=args.resolution_input)
            feature = backbone(image)
            current_data.add_repr(classe_idx, feature)
        classe_idx += 1


def preprocess(img, dtype=np.float32, shape_input=(32, 32)):
    """
    Args:
        img(np.ndarray(h,w,c)) :
    """
    assert len(img.shape) == 3
    assert img.shape[-1] == 3
    # img=img.astype(dtype)
    img = cv2.resize(
        img, dsize=shape_input, interpolation=cv2.INTER_LINEAR
    )  # linear is faster than cubic

    if img.dtype != dtype:
        img = img.astype(dtype)
    img = img[None, :]
    return (img / 255 - np.array([0.485, 0.456, 0.406], dtype=dtype)) / np.array(
        [0.229, 0.224, 0.225], dtype=dtype
    )


# addr_cam = "rtsp://admin:brain2021@10.29.232.40"
# cap = cv2.VideoCapture(addr_cam)

# constant of the program
SCALE = 1
RES_OUTPUT = (1280, 720)  # resolution = (1280,720)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def launch_demo():
    """
    initialize the variable and launch the demo
    """
    backbone = get_model(args.backbone_specs)
    few_shot_model = FewShotModel(args.classifier_specs)

    # data holding variables
    possible_input = [str(i) for i in range(177, 185)]
    class_num = len(possible_input)
    current_data = DataFewShot(class_num)

    # program related constant
    do_inference = False
    do_registration = False
    do_reset = False
    prev_frame_time = time.time()

    
    # time related variables
    clock = 0
    clock_m = 0
    clock_init = 20

    # CV2 related constant
    if not (args.camera_specification is None):
        cap = cv2.VideoCapture(args.camera_specification)

    # cv_interface manage graphical manipulation
    cv_interface = OpencvInterface(cap, SCALE, RES_OUTPUT, FONT, class_num)
    if  args.button_keyboard == "button" :
        from Input_Output.BoutonsManager import BoutonsManager
        btn_manager=BoutonsManager(args.overlay.btns_gpio)
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter("output.avi", fourcc, 30.0, RES_OUTPUT)

    number_image = 1

    while True:
        t = time.perf_counter()

        new_frame_time = time.time()
        # print('clock: ', clock)
        fps = int(1 / (new_frame_time - prev_frame_time))

        # stop simulation if max number of frame is attained
        if (
            not (args.max_number_of_frame is None)
            and number_image > args.max_number_of_frame
        ):
            break

        try:
            cv_interface.read_frame()
            print(f"reading image n°{number_image}")
            print(f"fps : {fps}")
            number_image = number_image + 1
        except:
            print("failed to get next image")
            break

        prev_frame_time = new_frame_time

        
        if args.button_keyboard=="keyboard" :
            key = cv_interface.get_key()
            key = chr(key)  # key convertion to char
        elif args.button_keyboard == "button" :
            key = btn_manager.change_state()
        else :
            print("L'argument button_keyboard n'est pas valide")
        
        if clock_m <= clock_init:
            frame = cv_interface.get_image()
            frame = preprocess(frame, shape_input=args.resolution_input)
            features = backbone(frame)

            current_data.add_mean_repr(features)
            if clock_m == clock_init:
                current_data.aggregate_mean_rep()
                if args.use_saved_sample:
                    path_sample = args.path_shots_video
                    compute_and_add_feature_saved_image(
                        backbone, cv_interface, current_data, path_sample
                    )
                    key = "i"  # simulate press of the key for inference

            cv_interface.put_text("Initialization")
            clock_m += 1

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

            frame = cv_interface.get_image()

            if key in possible_input:
                # print("saving snapshot of class", classe)
                cv_interface.add_snapshot(classe)

            # add the representation to the class
            frame = preprocess(frame, shape_input=args.resolution_input)
            features = backbone(frame)
            current_data.add_repr(classe, features)

            if abs(clock - last_detected) < 10:
                do_registration = True
                text = f"Class :{classe} registered. \
                Number of shots: {cv_interface.get_number_snapshot(classe)}"
                cv_interface.put_text(text)
            else:
                do_registration = False

        # reset action
        if key == 'r':
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

        # inference action
        if key == 'i' and current_data.is_data_recorded():
            print("doing inference")
            do_inference = True
            probabilities = None

        # perform inference
        if do_inference and clock_m > clock_init and not do_reset:

            frame = cv_interface.get_image()

            frame = preprocess(frame, shape_input=args.resolution_input)
            features = backbone(frame)
            classe_prediction, probabilities = few_shot_model.predict_class_moving_avg(
                features,
                probabilities,
                current_data.get_shot_list(),
                current_data.get_mean_features(),
            )

            # print("probabilities after exp moving average:", probabilities)
            cv_interface.put_text(f"Object is from class :", classe_prediction)
            # f'Probabilities :{list(map(lambda x:np.round(x, 2), probabilities.tolist()))}'
            cv_interface.draw_indicator(probabilities)

            if args.no_display and not (args.save_video):

                print("probabilities :", probabilities)

        # interface
        cv_interface.put_text(f"fps:{fps}", bottom_pos_x=0.05, bottom_pos_y=0.1)
        cv_interface.put_text(f"clock:{clock}", bottom_pos_x=0.8, bottom_pos_y=0.1)

        if not (args.no_display):
            cv_interface.show()

        if args.save_video:
            frame_to_save = cv_interface.frame
            out.write(frame_to_save)
        clock += 1

        if key == 'q':
            break

    cv_interface.close()
    if args.save_video:
        out.release()


launch_demo()
