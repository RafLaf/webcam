import cv2
import numpy as np
import torch
from torchvision import transforms, datasets
from resnet12 import ResNet12
import time
import torch.nn.functional as F

addr_cam = "rtsp://admin:brain2021@10.29.232.40"
device = 'cuda:0'

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
model = ResNet12(64, [3, 84, 84], 351, True, False).to(device)
#model = ResNet12(64, [3, 84, 84], 64, True, False).to(device)

def load_model_weights(model, path, device):
    pretrained_dict = torch.load(path, map_location=device)
    model_dict = model.state_dict()
    #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    new_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict:
            if 'bn' in k:
                new_dict[k] = v
            else:
                new_dict[k] = v.half()
    model_dict.update(new_dict) 
    model.load_state_dict(model_dict)
    print('Model loaded!')

#model.load_state_dict(torch.load('/home/r21lafar/Documents/dataset/mini1.pt1', map_location=device))
#model.load_state_dict(torch.load('/hdd/data/backbones/easybackbones/tieredlong1.pt1', map_location=device))
#model.load_state_dict(torch.load('/hdd/data/backbones/easybackbones/mini1.pt1', map_location=device))
load_model_weights(model, '/hdd/data/backbones/easybackbones/tieredlong1.pt1', device)

#mean_base_features = torch.load('/ssd2/data/AugmentedSamples/features/miniImagenet/AS600Vincent/mean_base3.pt', map_location=device).unsqueeze(0)
shots_list = []
registered_classes = []
shot_frames = []
#cap = cv2.VideoCapture(addr_cam)
cap = cv2.VideoCapture(0)
scale = 1
clock = 0
inference = False
registration = False
prev_frame_time = time.time()
font = cv2.FONT_HERSHEY_SIMPLEX

def draw_indicator(frame, percentages, shot_frames):
    def percentage_to_color(p):
        return 0,255 - (255 * p), 255 * p
    height, width, _ = frame.shape
    # config
    levels = 50
    level_width = width //10
    level_height = 5
    shift_y = int(height*0.4)
    # draw
    
    #cv2.rectangle(img, (10, img.shape[0] - (indicator_height + 10)), (10 + indicator_width, img.shape[0] - 10), (0, 0, 0), cv2.FILLED)
    cv2.rectangle(frame, (20  , shift_y - level_height * (levels+10) ), (20 + level_width*(percentages.shape[0]-1) + level_width -10,shift_y - level_height * (levels+1)  ) ,(0, 0, 0), cv2.FILLED)
    cv2.rectangle(frame, (20  , shift_y + level_height * 1 ), (20 + level_width*(percentages.shape[0]-1) + level_width -10,shift_y + level_height * 10  ) ,(0, 0, 0), cv2.FILLED)
    for k in range(percentages.shape[0]):
        images = shot_frames[k]
        s = images[0].shape
        #frame[20 + level_width*k :20 + level_width*k +s[0] , shift_y + level_height * 10:shift_y + level_height * 10+s[1]] = image
        y_start_img = shift_y 
        x_start_img = 15+level_width*k  
        for n_shot in range(len(images)):
            if y_start_img + s[0] + n_shot*(s[0]+10)< frame.shape[0]:
                frame[ y_start_img + n_shot*(s[0]+10):y_start_img + s[0] + n_shot*(s[0]+10), x_start_img:x_start_img + s[1]] = images[n_shot] #.reshape(s[1], s[0], -1)

        img_level = int(percentages[k] * levels)
        #cv2.putText(frame, str(np.round(percentages[k].item(),2)*100)+'%', (20 + level_width*k  , shift_y - level_height * (levels+3)), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f'{np.round(100*percentages[k].item(), 2)}%', (20 + level_width*k  , shift_y - level_height * (levels+3)), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.rectangle(frame,(20 + level_width*k , shift_y - levels* level_height), (20 + level_width*k + level_width -10,shift_y  ) , (0,0,0), cv2.FILLED)
        for i in range(img_level):
            level_y_b = shift_y - i * level_height
            start_point = (20 + level_width*k , level_y_b - level_height)
            end_point =  (20 + level_width*k + level_width -10 , level_y_b)
            cv2.rectangle(frame, start_point, end_point , percentage_to_color(i / levels), cv2.FILLED)
            #cv2.rectangle(frame,start_point, end_point, percentage_to_color(i / levels), cv2.FILLED)
            #if i==0:
            #    cv2.putText(frame, str(k), (end_point[0] -level_width//2, end_point[1]+40), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

clock_M = 0
clock_init = 20
mean_features = []
#resolution = (1280,720)
resolution = (1920,1080)
resetting = False
K_nn = 5
model_name = 'knn'

def predict(shots_list, features, model_name):
    if model_name == 'ncm':
        shots = torch.stack([s.mean(dim=0) for s in shots_list])
        distances = torch.norm(shots-features, dim = 1, p=2)
        classe_prediction = distances.argmin().item()
        probas = F.softmax(-20*distances, dim=0).detach().cpu()
    elif model_name == 'knn':
        shots = torch.cat(shots_list)
        #create target list of the shots
        targets = torch.cat([torch.Tensor([i]*shots_list[i].shape[0]) for i in range(len(shots_list))])
        distances = torch.norm(shots-features, dim = 1, p=2)
        #get the k nearest neighbors

        _, indices = distances.topk(K_nn, largest=False)
        probas = F.one_hot(targets[indices].long(), num_classes=len(shots_list)).sum(dim=0)/K_nn
        classe_prediction = probas.argmax().item()
    return probas, classe_prediction

while(True):
    ret,frame = cap.read()
    frame = cv2.resize(frame, resolution, interpolation = cv2.INTER_AREA)
    height, width, _ = frame.shape
    new_frame_time = time.time()
    #print('clock: ', clock)    
    fps = int(1/(new_frame_time-prev_frame_time))
    prev_frame_time = new_frame_time
    if clock_M <= clock_init:
        img = apply_transformations(frame).to(device)
        _, features = model(img.unsqueeze(0))
        mean_features.append(features.detach().to(device))
        if clock_M == clock_init:
            mean_features = torch.cat(mean_features, dim = 0)
            mean_features = mean_features.mean(dim = 0)
        cv2.putText(frame, f'Initialization', (int(width*0.4), int(height*0.1)), font, scale, (255, 0, 0), 3, cv2.LINE_AA)

        clock_M += 1        

    key = cv2.waitKey(33) & 0xFF
    # shot acquisition
    if (key in range(48, 53) or registration) and clock_M>clock_init and not resetting:
        #if key in range(48, 53):
        registration = True
        inference = False
        
        if key in range(48, 53):
            classe = key-48
            last_detected = clock*1 #time.time()
        print('class :', classe)
        
        img = apply_transformations(frame).to(device)
        _, features = model(img.unsqueeze(0))
        # preprocess features
        features = preprocess(features, mean_base_features= mean_features)
        print('features:', features.shape)
        image_label = cv2.resize(frame, (int(frame.shape[1]//10),int(frame.shape[0]//10 )), interpolation = cv2.INTER_AREA)
        if classe not in registered_classes:
            registered_classes.append(classe)
            shots_list.append(features)
            if key in range(48, 53):
                shot_frames.append([image_label])
        else:
            shots_list[classe] = torch.cat((shots_list[classe], features), dim = 0)
            print('------------:', shots_list[classe].shape)
            if key in range(48, 53):
                shot_frames[classe].append(image_label)

    if registration:
        if abs(clock-last_detected)<10 and inference==False:
            cv2.putText(frame, f'Class :{classe} registered. Number of shots: {len(shot_frames[classe])}', (int(width*0.4), int(height*0.1)), font, scale, (255, 0, 0), 3, cv2.LINE_AA)
        else:
            registration = False

    if key == ord('r'):
        registration = False
        inference = False
        shots_list = []
        shot_frames = []
        registered_classes = []
        reset_clock = 0
        resetting  = True
        
    if resetting:
        cv2.putText(frame, f'Reset', (int(width*0.4), int(height*0.1)), font, scale, (255, 0, 0), 3, cv2.LINE_AA)
        reset_clock += 1
        if reset_clock > 20:
            resetting = False

    if key == ord('i') and len(shots_list)>0:
        inference = True
        probabilities = None

    if inference and clock_M>clock_init and not resetting:
        img = apply_transformations(frame).to(device)
        _, features = model(img.unsqueeze(0))
        features = preprocess(features, mean_base_features= mean_features)
        probas, _ = predict(shots_list, features, model_name=model_name)
        print('probabilities:', probas)
        if probabilities == None:
            probabilities = probas
        else:
            if model_name == 'ncm':
                probabilities = probabilities*0.85 + probas*0.15
            elif model_name == 'knn':
                probabilities = probabilities*0.95 + probas*0.05
        classe_prediction = probabilities.argmax().item()
        print('probabilities after exp moving average:', probabilities)
        cv2.putText(frame, f'Object is from class :{classe_prediction}', (int(width*0.4), int(height*0.1)), font, scale, (255, 0, 0), 3, cv2.LINE_AA)
        #cv2.putText(frame, f'Probabilities :{list(map(lambda x:np.round(x, 2), probabilities.tolist()))}', (7, 750), font, 3, (255, 0, 0), 3, cv2.LINE_AA)
        draw_indicator(frame,probabilities, shot_frames)
    cv2.putText(frame, f'fps:{fps}', (int(width*0.05), int(height*0.1)), font, scale, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, f'clock:{clock}', (int(width*0.8), int(height*0.1)), font, scale, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('frame',frame)
    clock += 1
    # reset clock
    #if clock == 100: clock = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()