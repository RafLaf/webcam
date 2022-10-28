
print("importing cv2")
import cv2
import numpy as np
print("importing torch")
import torch

import time
import copy


from utils import opencv_interface
from possible_models import get_model,load_model_weights,predict_class_moving_avg
from preprocess import image_preprocess,feature_preprocess

print("import done")

#addr_cam = "rtsp://admin:brain2021@10.29.232.40"
device = 'cuda:0'

# 1, 2, 3... for every class we're adding
# i for starting inference, it will be run every 1 second
# q for exiting the program

# Get the model

model_specs={
    "feature_maps":64, 
    "input_shape":[3,84,84],
    "num_classes":351, 
    "few_shot":True, 
    "rotations":False
}

model=get_model("resnet12",model_specs).to(device)
#model = ResNet12(64, [3, 84, 84], 351, True, False).to(device)
#model = ResNet12(64, [3, 84, 84], 64, True, False).to(device)



def save_feature(data,classe,features):
    if classe not in data["registered_classes"]:
        data["registered_classes"].append(classe)
        data["shot_list"].append(features)
    else:
        data["shot_list"][classe] = torch.cat((data["shot_list"][classe], features), dim = 0)
        print('------------:', data["shot_list"][classe].shape)






#model.load_state_dict(torch.load('/home/r21lafar/Documents/dataset/mini1.pt1', map_location=device))
#model.load_state_dict(torch.load('/hdd/data/backbones/easybackbones/tieredlong1.pt1', map_location=device))
#model.load_state_dict(torch.load('/hdd/data/backbones/easybackbones/mini1.pt1', map_location=device))

path_model="weight/tieredlong1.pt1"
load_model_weights(model, path_model, device)

#mean_base_features = torch.load('/ssd2/data/AugmentedSamples/features/miniImagenet/AS600Vincent/mean_base3.pt', map_location=device).unsqueeze(0)

#cap = cv2.VideoCapture(addr_cam)

#CV2 related constant
cap = cv2.VideoCapture(0)
scale = 1
resolution_output = (1920,1080)#resolution = (1280,720)
font = cv2.FONT_HERSHEY_SIMPLEX

cv_interface=opencv_interface(cap,scale,resolution_output,font)

#program related constant
do_inference = False
do_registration = False
do_reset = False
prev_frame_time = time.time()

possible_input=[i for i in range(48, 53)]

#data holding variables
empty_data={
    
    "registered_classes":[],
    "shot_frames":[[] for i in range(len(possible_input))],
    "shot_list":[],
    "mean_features" : []
}
data=copy.deepcopy(empty_data)


#time related variables
clock = 0
clock_M = 0
clock_init = 20

#model parameters
model_name = 'knn'



while(True):
    cv_interface.read_frame()
    
    new_frame_time = time.time()
    #print('clock: ', clock)    
    fps = int(1/(new_frame_time-prev_frame_time))
    prev_frame_time = new_frame_time
    
    if clock_M <= clock_init:
        frame=cv_interface.get_image()
        img = image_preprocess(frame).to(device)
        _, features = model(img.unsqueeze(0))
        data["mean_features"].append(features.detach().to(device))
        if clock_M == clock_init:
            data["mean_features"] = torch.cat(data["mean_features"], dim = 0)
            data["mean_features"] = data["mean_features"].mean(dim = 0)

        cv_interface.put_text(f'Initialization')

        clock_M += 1        

    key = cv2.waitKey(33) & 0xFF
    
    # shot acquisition
    if (key in possible_input or do_registration) and clock_M>clock_init and not do_reset:
        do_inference = False
        
        if key in possible_input:
            classe = possible_input.index(key)
            last_detected = clock*1 #time.time()

        print('class :', classe)
        frame=cv_interface.get_image()
        img = image_preprocess(frame).to(device)
        _, features = model(img.unsqueeze(0))

        # preprocess features
        features = feature_preprocess(features, mean_base_features= data["mean_features"])
        print('features:', features.shape)
        if key in possible_input:
            print(f"saving snapshot of class {classe}")
            cv_interface.add_snapshot(data,classe)
        #add the representation to the class
        
        save_feature(data,classe,features)
        if abs(clock-last_detected)<10:
            do_registration = True
            text=f'Class :{classe} registered. Number of shots: {len(data["shot_frames"][classe])}'
            cv_interface.put_text(text)
        else:
            do_registration = False
    
    #reset action
    if key == ord('r'):
        do_registration = False
        do_inference = False
        mean_features=data["mean_features"]
        data=copy.deepcopy(empty_data)
        data["mean_features"]=mean_features
        reset_clock = 0
        do_reset  = True
        
    if do_reset:
        cv_interface.put_text("Resnet background inference")
        reset_clock += 1
        if reset_clock > 20:
            do_reset = False
    
    #inference actionfont
    if key == ord('i') and len(data["shot_list"])>0:
        do_inference = True
        probabilities = None
    
    #perform infernece
    if do_inference and clock_M>clock_init and not do_reset:
        frame= cv_interface.get_image()
        img = image_preprocess(frame).to(device)
       
        classe_prediction,probabilities=predict_class_moving_avg(img,data,model,model_name,probabilities)
        
        print('probabilities after exp moving average:', probabilities)
        cv_interface.put_text(f'Object is from class :{classe_prediction}')
        #f'Probabilities :{list(map(lambda x:np.round(x, 2), probabilities.tolist()))}'
        cv_interface.draw_indicator(probabilities,data["shot_frames"])

    #interface
    cv_interface.put_text(f"fps:{fps}",bottom_pos_x=0.05,bottom_pos_y=0.1)
    cv_interface.put_text(f"clock:{clock}",bottom_pos_x=0.8,bottom_pos_y=0.1)
    cv_interface.show()
    
    clock += 1
    # reset clock
    #if clock == 100: clock = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv_interface.close()