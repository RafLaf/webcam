
print("importing cv2")
import cv2
import numpy as np
print("importing torch")
import torch
from torchvision import transforms, datasets
from resnet12 import ResNet12
import time
import torch.nn.functional as F
from utils import opencv_interface
import copy

print("import done")

#addr_cam = "rtsp://admin:brain2021@10.29.232.40"
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

def save_feature(data,classe,features):
    if classe not in data["registered_classes"]:
        data["registered_classes"].append(classe)
        data["shot_list"].append(features)
    else:
        data["shot_list"][classe] = torch.cat((data["shot_list"][classe], features), dim = 0)
        print('------------:', data["shot_list"][classe].shape)



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

def predict_class_moving_avg(img,data,model_name,probabilities):
     
    _, features = model(img.unsqueeze(0))
    
    features = feature_preprocess(features, mean_base_features= data["mean_features"])
    
    probas, _ = predict(data["shot_list"], features, model_name=model_name)
    print('probabilities:', probas)
    
    if probabilities == None:
        probabilities = probas
    else:
        if model_name == 'ncm':
            probabilities = probabilities*0.85 + probas*0.15
        elif model_name == 'knn':
            probabilities = probabilities*0.95 + probas*0.05

    classe_prediction = probabilities.argmax().item()
    return classe_prediction,probabilities


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
K_nn = 5
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
       
        classe_prediction,probabilities=predict_class_moving_avg(img,data,model_name,probabilities)
        
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