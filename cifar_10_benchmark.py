import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

from possible_models import get_model,load_model_weights, get_features
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


trainset,testset,classes=get_loader()


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

TRANSFORMS = transforms.Compose(
        [transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#other constant
NUMBER_SHOT=1
NUMBER_CLASS=10

# model related
backbone = get_model("resnet12", MODEL_SPECS).to(DEVICE)
load_model_weights(backbone, PATH_MODEL, device=DEVICE)


data=create_data(trainset,backbone,TRANSFORMS)

#predict_feature()