import cv2
import numpy as np
import torch
from torchvision import transforms, datasets
from resnet12 import ResNet12
import time
import torch.nn.functional as F
from utils import draw_indicator
addr_cam = "rtsp://admin:brain2021@10.29.232.40"
device = 'cuda:0'

# 1, 2, 3... for every class we're adding
# i for starting inference, it will be run every 1 second
# q for exiting the program

# Apply transformations
def image_preprocess(img):
    img = transforms.ToTensor()(img)
    norm = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    all_transforms = torch.nn.Sequential(transforms.Resize(110), transforms.CenterCrop(100), norm)
    img = all_transforms(img)
    return img

def feature_preprocess(features, mean_base_features=None):
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

            #bn : keep precision (low cost associated)
            #does this work for the fpga ?
            if 'bn' in k:
                new_dict[k] = v
            else:
                new_dict[k] = v.to(torch.float16)
    model_dict.update(new_dict) 
    model.load_state_dict(model_dict)
    print('Model loaded!')

#model.load_state_dict(torch.load('/home/r21lafar/Documents/dataset/mini1.pt1', map_location=device))
#model.load_state_dict(torch.load('/hdd/data/backbones/easybackbones/tieredlong1.pt1', map_location=device))
#model.load_state_dict(torch.load('/hdd/data/backbones/easybackbones/mini1.pt1', map_location=device))
load_model_weights(model, '/hdd/data/backbones/easybackbones/tieredlong1.pt1', device)

#mean_base_features = torch.load('/ssd2/data/AugmentedSamples/features/miniImagenet/AS600Vincent/mean_base3.pt', map_location=device).unsqueeze(0)

#cap = cv2.VideoCapture(addr_cam)

#CV2 related constant
cap = cv2.VideoCapture(0)
scale = 1

#program related constant
do_inference = False
do_registration = False
do_reset = False
prev_frame_time = time.time()
font = cv2.FONT_HERSHEY_SIMPLEX

#data holding variables
shots_list = []
registered_classes = []
shot_frames = []
mean_features = []

#time related variables
clock = 0
clock_M = 0
clock_init = 20

#model parameters
resolution = (1920,1080)#resolution = (1280,720)
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
        probas = F.one_hot(targets[indices].to(torch.int64), num_classes=len(shots_list)).sum(dim=0)/K_nn
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
        img = image_preprocess(frame).to(device)
        _, features = model(img.unsqueeze(0))
        mean_features.append(features.detach().to(device))
        if clock_M == clock_init:
            mean_features = torch.cat(mean_features, dim = 0)
            mean_features = mean_features.mean(dim = 0)
        cv2.putText(frame, f'Initialization', (int(width*0.4), int(height*0.1)), font, scale, (255, 0, 0), 3, cv2.LINE_AA)

        clock_M += 1        

    key = cv2.waitKey(33) & 0xFF
    # shot acquisition

    
    if (key in range(48, 53) or do_registration) and clock_M>clock_init and not do_reset:
        #if key in range(48, 53):
        do_registration = True
        do_inference = False
        
        if key in range(48, 53):
            classe = key-48
            last_detected = clock*1 #time.time()
        print('class :', classe)
        
        img = image_preprocess(frame).to(device)
        _, features = model(img.unsqueeze(0))
        # preprocess features
        features = feature_preprocess(features, mean_base_features= mean_features)
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

    if do_registration:
        if abs(clock-last_detected)<10 and not do_inference:
            cv2.putText(frame, f'Class :{classe} registered. Number of shots: {len(shot_frames[classe])}', (int(width*0.4), int(height*0.1)), font, scale, (255, 0, 0), 3, cv2.LINE_AA)
        else:
            do_registration = False

    if key == ord('r'):
        do_registration = False
        do_inference = False
        shots_list = []
        shot_frames = []
        registered_classes = []
        reset_clock = 0
        do_reset  = True
        
    if do_reset:
        cv2.putText(frame, f'Reset', (int(width*0.4), int(height*0.1)), font, scale, (255, 0, 0), 3, cv2.LINE_AA)
        reset_clock += 1
        if reset_clock > 20:
            do_reset = False

    if key == ord('i') and len(shots_list)>0:
        do_inference = True
        probabilities = None

    if do_inference and clock_M>clock_init and not do_reset:
        img = image_preprocess(frame).to(device)
        _, features = model(img.unsqueeze(0))
        features = feature_preprocess(features, mean_base_features= mean_features)
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
        draw_indicator(frame,probabilities, shot_frames,font,scale)
    
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