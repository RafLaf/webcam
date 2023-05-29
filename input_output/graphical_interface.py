"""
manage the graphical interface and camera for the demo
"""
import cv2
import numpy as np


def draw_indic(frame, percentages, shot_frames, font, scale):
    """
    Args :
        percentages : (np.ndarray(1,n_features) ) : probability of belonging to each class
    """
    percentages = percentages[0]  # not clean,

    def percentage_to_color(p):
        return 0, 255 - (255 * p), 255 * p

    height, width, _ = frame.shape
    # config
    levels = 50
    level_width = width // 10
    level_height = 5
    shift_y = int(height * 0.4)
    # draw

    cv2.rectangle(
        frame,
        (20, shift_y - level_height * (levels + 10)),
        (
            20 + level_width * (percentages.shape[0] - 1) + level_width - 10,
            shift_y - level_height * (levels + 1),
        ),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.rectangle(
        frame,
        (20, shift_y + level_height * 1),
        (
            20 + level_width * (percentages.shape[0] - 1) + level_width - 10,
            shift_y + level_height * 10,
        ),
        (0, 0, 0),
        cv2.FILLED,
    )
    for k in range(percentages.shape[0]):
        images = shot_frames[k]
        s = images[0].shape
        y_start_img = shift_y
        x_start_img = 15 + level_width * k
        for n_shot in range(len(images)):
            if y_start_img + s[0] + n_shot * (s[0] + 10) < frame.shape[0]:
                frame[
                    y_start_img
                    + n_shot * (s[0] + 10) : y_start_img
                    + s[0]
                    + n_shot * (s[0] + 10),
                    x_start_img : x_start_img + s[1],
                ] = images[n_shot]
        img_level = int(percentages[k] * levels)
        cv2.putText(
            frame,
            f"{np.round(100*percentages[k].item(), 2)}%",
            (20 + level_width * k, shift_y - level_height * (levels + 3)),
            font,
            scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.rectangle(
            frame,
            (20 + level_width * k, shift_y - levels * level_height),
            (20 + level_width * k + level_width - 10, shift_y),
            (0, 0, 0),
            cv2.FILLED,
        )
        for i in range(img_level):
            level_y_b = shift_y - i * level_height
            start_point = (20 + level_width * k, level_y_b - level_height)
            end_point = (20 + level_width * k + level_width - 10, level_y_b)
            cv2.rectangle(
                frame,
                start_point,
                end_point,
                percentage_to_color(i / levels),
                cv2.FILLED,
            )


class OpencvInterface:
    """
    Class representing the opencv configuration
    (Manage the camera and the graphical interface)
    this class also has the frame attribute since it needs ownership (to modify it)
    once the image is modified, you can no longer access it
    ...

    Attributes :
        video_capture(cv.VideoCapture) : camera used
        scale : scale of the indicators objects
        height : height of the interface
        width : width of the interface
        resolution : height, width of the interface
        font : font used by opencv
        frame : current captured frame
        number_of_class : number of possible class
        snapshot : saved snapshots
        is_available_frame : weither the frame is available for capture


    """

    def __init__(self, video_capture, scale, resolution, font, number_of_class):
        self.video_capture = video_capture
        self.scale = scale
        self.height = resolution[0]
        self.width = resolution[1]
        self.resolution = resolution
        self.font = font
        self.frame = None
        self.number_of_class = number_of_class
        self.snapshot = [[] for i in range(number_of_class)]
        self.is_present_original_frame = False

    def read_frame(self):
        """
        read and resize the frame to interface size
        """
        _, frame = self.video_capture.read()
        self.frame = cv2.resize(frame, self.resolution, interpolation=cv2.INTER_AREA)
        self.is_present_original_frame = True

    def get_copy_captured_image(self, resolution):
        """
        return a resized copy of the captured image if it still present in the data
        """

        if self.is_present_original_frame:
            return cv2.resize(
                self.frame, dsize=resolution, interpolation=cv2.INTER_LINEAR
            )  # linear is faster than cubic
        else:
            raise Exception("original frame is not available")

    def put_text(self, text, bottom_pos_x=0.4, bottom_pos_y=0.1):
        """
        put some text in the inteface
            parameters :
                text(string) : text to be added
                bottom_pos_x(float) : x position of the bottom left pixel (% of the whole frame)
                bottom_pos_y(float) : y position of the bottom left pixel (% of the whole frame)
        """
        self.is_present_original_frame = False
        cv2.putText(
            self.frame,
            text,
            (int(self.width * bottom_pos_x), int(self.height * bottom_pos_y)),
            self.font,
            self.scale,
            (255, 0, 0),
            3,
            cv2.LINE_AA,
        )

    def show(self):
        """
        show the current updated frame
        """
        cv2.imshow("frame", self.frame)

    def draw_indicator(self, probabilities):
        """
        wrapper of draw_indic
        """
        self.is_present_original_frame = False
        draw_indic(self.frame, probabilities, self.snapshot, self.font, self.scale)

    def add_snapshot(self, classe, frame_to_add=None):
        """
        add a snapshot to memmory
        """
        if frame_to_add is None:
            frame_to_add = self.frame
        image_label = cv2.resize(
            frame_to_add,
            (int(self.height // 10), int(self.width // 10)),
            interpolation=cv2.INTER_AREA,
        )
        self.snapshot[classe].append(image_label)

    def get_number_snapshot(self, classe):
        """
        get the number of snapshot of a given classe"""
        return len(self.snapshot[classe])

    def reset_snapshot(self):
        """
        reset the snapshot to initial value"""
        self.snapshot = [[] for i in range(self.number_of_class)]

    def close(self):
        """
        liberate all attributed ressources
        """
        self.video_capture.release()
        cv2.destroyAllWindows()

    def get_key(self):
        """
        if a key was pressed, get the key
        """
        return cv2.waitKey(33) & 0xFF
