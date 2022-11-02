import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

from possible_models import get_model,load_model_weights, get_features,predict_feature
from data_few_shot import DataFewShot

def get_loader():
    """
    get the loader, see : 
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

    """

    #batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                        shuffle=False, num_workers=1)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainset,testset,classes

def create_data(trainset,backbone,transform,shot_number=1,num_class=10):
    """
    create a few_shot data
    """
    data=DataFewShot(num_class)
    number_sample=np.zeros(num_class)
    total_sample=len(trainset)
    iteration=0

    while not(np.all(number_sample==shot_number)) and iteration<total_sample:
        img,classe=trainset[iteration]
        if number_sample[classe]<shot_number:
            print(f"adding one sample to class {classe}")
            repr=get_features(img,backbone,DEVICE,transform=transform)
            data.add_repr(classe,repr)
            data.add_mean_repr(repr,DEVICE)
            number_sample[classe]+=1
        
        iteration+=1
    data.aggregate_mean_rep()
    return data




def get_performance(test_set,data,backbone,CLASSIFIER_SPECS,transform=None):
    correct_pred=0
    incorect_pred=0

    list_pred=[]
    list_gt=[]
    for i,(img,gt_classe) in enumerate(test_set):
        
        if i==100:
            break

        repr=get_features(img,backbone,DEVICE,transform=transform)
    
        classe_pred,prediction=predict_feature(data.shot_list,
            repr,
            data.mean_features,
            CLASSIFIER_SPECS["model_name"],
            **CLASSIFIER_SPECS["args"])

        list_pred.append(classe_pred)
        list_gt.append(gt_classe)

        if classe_pred==gt_classe:
            correct_pred+=1
        else:
            incorect_pred+=1
    return correct_pred/(correct_pred+incorect_pred),list_pred,list_gt
            


trainset,testset,classes=get_loader()


MODEL_SPECS = {
    "feature_maps": 64,
    "input_shape": [3, 32, 32],
    "num_classes": 64,  # 351
    "few_shot": True,
    "rotations": False,
}
PATH_MODEL = "weight/cifar1.pt1"

# model parameters
CLASSIFIER_SPECS = {
    "model_name":"knn",
    "args":{
        "number_neighboors":1
    }
}
DEVICE = "cuda:0"

TRANSFORMS = transforms.Compose(
        [transforms.ToTensor(), 
        transforms.Normalize(
            [0.5,0.5,0.5],# np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
            [0.5,0.5,0.5]# np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]),
        ),
        transforms.Resize(32)])

#number to save
NUMBER_SHOT=4

# model related
backbone = get_model("resnet12", MODEL_SPECS).to(DEVICE)
load_model_weights(backbone, PATH_MODEL, device=DEVICE)

data=create_data(trainset,backbone,TRANSFORMS,shot_number=NUMBER_SHOT)

print(get_performance(testset,data,backbone,CLASSIFIER_SPECS,transform=TRANSFORMS))