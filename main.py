import cv2
import numpy as np
import torch
from torchvision import transforms, datasets
from resnet12 import ResNet12
import time
import torch.nn.functional as F

addr_cam = "rtsp://admin:brain2021@10.29.232.41"
device = 'cpu'


# 1, 2, 3... for every class we're adding
# i for starting inference, it will be run every 1 second
# q for exiting the program

# Apply transformations
def apply_transformations(img):
    img = transforms.ToTensor()(img)
    norm = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    all_transforms = torch.nn.Sequential(transforms.Resize(110), transforms.CenterCrop(100), norm)
    img = all_transforms(img)
    return img

def preprocess(features, mean_base_features=None):
    features = features - mean_base_features
    features = features / torch.norm(features, dim = 1, keepdim = True)
    return features

# Get the model
#model = ResNet12(64, [3, 84, 84], 351, True, False).to(device)
model = ResNet12(64, [3, 84, 84], 64, True, False).to(device)

model.load_state_dict(torch.load('/home/r21lafar/Documents/dataset/mini1.pt1', map_location=device))
#model.load_state_dict(torch.load('/hdd/data/backbones/easybackbones/tieredlong1.pt1', map_location=device))

#mean_base_features = torch.load('/ssd2/data/AugmentedSamples/features/miniImagenet/AS600Vincent/mean_base3.pt', map_location=device).unsqueeze(0)
shots_list = []
registered_classes = []
shot_frames = []
cap = cv2.VideoCapture(addr_cam)
clock = 0
inference = False
registration = False
prev_frame_time = time.time()
font = cv2.FONT_HERSHEY_SIMPLEX

def draw_indicator(frame, percentages, shot_frames):
    def percentage_to_color(p):
        return 0,255 - (255 * p), 255 * p

    # config
    levels = 50
    level_width = 100
    level_height = 5
    shift_y = 1200
    # draw
    
    #cv2.rectangle(img, (10, img.shape[0] - (indicator_height + 10)), (10 + indicator_width, img.shape[0] - 10), (0, 0, 0), cv2.FILLED)
    cv2.rectangle(frame, (20  , shift_y - level_height * (levels+10) ), (20 + level_width*(percentages.shape[0]-1) + level_width -10,shift_y - level_height * (levels+1)  ) ,(0, 0, 0), cv2.FILLED)
    cv2.rectangle(frame, (20  , shift_y + level_height * 1 ), (20 + level_width*(percentages.shape[0]-1) + level_width -10,shift_y + level_height * 10  ) ,(0, 0, 0), cv2.FILLED)
    for k in range(percentages.shape[0]):
        image = shot_frames[k]
        s = image.shape
        print('shape:', s)
        #frame[20 + level_width*k :20 + level_width*k +s[0] , shift_y + level_height * 10:shift_y + level_height * 10+s[1]] = image
        frame[20 + level_width :20 + level_width +s[0] , shift_y + level_height*k :shift_y + level_height*k + s[1]] = image

        img_level = int(percentages[k] * levels)
        cv2.putText(frame, str(np.round(percentages[k].item(),2)*100)+'%', (20 + level_width*k  , shift_y - level_height * (levels+3)), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.rectangle(frame,(20 + level_width*k , shift_y - levels* level_height), (20 + level_width*k + level_width -10,shift_y  ) , (0,0,0), cv2.FILLED)
        for i in range(img_level):
            level_y_b = shift_y - i * level_height
            start_point = (20 + level_width*k , level_y_b - level_height)
            end_point =  (20 + level_width*k + level_width -10 , level_y_b)
            #cv2.rectangle(img, start_point, end_point , percentage_to_color(i / levels), cv2.FILLED)
            cv2.rectangle(frame,start_point, end_point, percentage_to_color(i / levels), cv2.FILLED)
            if i==0:
                cv2.putText(frame, str(k), (end_point[0] -level_width//2, end_point[1]+40), font, 1, (255, 255, 255), 1, cv2.LINE_AA)


while(True):
    ret,frame = cap.read()
    height, width, _ = frame.shape
    new_frame_time = time.time()
    #print('clock: ', clock)    
    fps = int(1/(new_frame_time-prev_frame_time))
    prev_frame_time = new_frame_time
    
    key = cv2.waitKey(33) & 0xFF
    
    if key in range(48, 53):
        registration = True
        inference = False
        
        last_detected = time.time()
        classe = key-48
        print('class :', classe)
        shot_frames.append(cv2.resize(frame, (int(frame.shape[0]//20),int(frame.shape[1]//20 )), interpolation = cv2.INTER_AREA))
        img = apply_transformations(frame).to(device)
        _, features = model(img.unsqueeze(0))
        # preprocess features
        #features = preprocess(features, mean_base_features)
        print('features:', features.shape)

        if classe not in registered_classes:
            registered_classes.append(classe)
            shots_list.append(features)
        else:
            shots_list[classe]= features

    if fps!=0 and clock % fps == 0 and inference:
        img = apply_transformations(frame).to(device)
        _, features = model(img.unsqueeze(0))
    
    if registration:
        if time.time()-last_detected<3:
            cv2.putText(frame, f'Class :{classe} registered', (7, 250), font, 3, (255, 0, 0), 3, cv2.LINE_AA)
        else:
            registration = False

    if key == ord('i'):
        inference = True
        probabilities = None
    if inference:

        shots = torch.cat(shots_list)
        print('shots:', shots.shape)
        img = apply_transformations(frame).to(device)
        _, features = model(img.unsqueeze(0))
        #features = preprocess(features, mean_base_features)
        distances = torch.norm(shots-features, dim = 1, p=2)
        prediction = distances.argmin().item()
        print('distances:', distances)
        probas = F.softmax(-20*distances).detach().cpu()
        if probabilities == None:
            probabilities = probas
        else:
            probabilities = probabilities*0.95 + probas*0.05
        print('pred:', prediction)
        print('probas:', probas)
        print('probabilities:', probabilities)
        cv2.putText(frame, f'Object is from class :{prediction}', (int(width*0.05), int(height*0.5)), font, 3, (255, 0, 0), 3, cv2.LINE_AA)
        #cv2.putText(frame, f'Probabilities :{list(map(lambda x:np.round(x, 2), probabilities.tolist()))}', (7, 750), font, 3, (255, 0, 0), 3, cv2.LINE_AA)
        draw_indicator(frame, probabilities, shot_frames)
    cv2.putText(frame, f'fps:{fps}', (int(width*0.05), int(height*0.1)), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, f'clock:{clock}', (int(width*0.8), int(height*0.1)), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('frame',frame)
    clock += 1
    # reset clock
    if clock == 100: clock = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()