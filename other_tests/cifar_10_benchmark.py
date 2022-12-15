"""
test the performance of the demo using the cifar10 dataset

Metric used to test : accuracy on the validation class (not a few-shot setting)

use torch for the metrics
"""

import torch
import torchvision
from torchvision import transforms
import numpy as np

from few_shot_model.few_shot_model import FewShotModel
from torch_evaluation.backbone_loader import get_model



def create_data(
    model,
    trainset,
    num_shots=1,
):
    """
    create a few_shot data
    """
    num_class=len(trainset.classes)
    #data_fewshot = DataFewShot(num_class)
    number_sample = np.zeros(num_class, dtype=np.int64)
    shots=None
    total_sample = len(trainset)
    iteration = 0
    print("total sample in training :", total_sample)

    while iteration < total_sample and (np.any(number_sample < num_shots)):
        img, classe = trainset[iteration]
        
        if number_sample[classe] < num_shots:
            representation = model(img)
            if iteration==0:
                feature_dim=representation.shape[-1]
                shots=np.zeros((num_class,num_shots,feature_dim))
            
            
            shots[classe,number_sample[classe],:]=representation
            number_sample[classe]+=1
           
        iteration += 1
    print("shots size: ",shots.shape )
    print("number of sample per class", number_sample)
    
    # print("mean repr",data_fewshot.mean_features)
    return shots


def get_performance(model, few_shot_model, shots, dataset):
    """
    get the performance of the model using the given few_shot setting
    """
    correct_pred = 0
    incorect_pred = 0

    list_pred = []
    list_gt = []

    mean_shots=np.mean(shots,axis=(0,1))
    for i, (img, gt_classe) in enumerate(dataset):

        if i == 1000:
            break

        features = model(img)
        classe_pred, _ = few_shot_model.predict_class_feature(
            features, 
            shots,
            mean_shots,
        )
        classe_pred=classe_pred [0]#delete leading dim
        

        list_pred.append(classe_pred)
        list_gt.append(gt_classe)

        if classe_pred == gt_classe:
            correct_pred += 1
        else:
            incorect_pred += 1

    return correct_pred / (correct_pred + incorect_pred), list_pred, list_gt


torch.manual_seed(0)

# model parameters


BACKBONE_SPECS = {
    "model_name": "resnet12",
    "path": "weight/cifar1.pt1",  # tieredlong1.pt1",
    "kwargs": {
        "feature_maps": 64,
        "input_shape": [3, 32, 32],
        "num_classes": 64,  # 351,
        "few_shot": True,
        "rotations": False,
    },
}

CLASSIFIER_SPECS = {"model_name": "ncm",}# "kwargs": {"number_neighboors": 1}}
#CLASSIFIER_SPECS = {"model_name": "knn", "kwargs": {"number_neighboors": 1}}

DEVICE = "cuda:0"

# TRANSFORMS = get_camera_preprocess()

TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),  # np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]),
        ),
        # transforms.Resize(42),
        # transforms.CenterCrop(32)
    ]
)
raise UserWarning("this file was not updated. neither in a few shot setting, and do not work")
# number to save
NUMBER_SHOT = 5

backbone = get_model(BACKBONE_SPECS, DEVICE)


# model related

TRAIN=torchvision.datasets.CIFAR10(root="./data", train=True, download=True,transform=TRANSFORMS)
TEST= torchvision.datasets.CIFAR10(root="./data", train=False, download=True,transform=TRANSFORMS)


#sub_train = torch.utils.data.Subset(TRAIN, list(range(100)))

    
shots = create_data(
    backbone, TRAIN, num_shots=NUMBER_SHOT
)
final_perf, a, b = get_performance(
    backbone, FewShotModel(CLASSIFIER_SPECS), shots, TEST
)  # final_list_pred, final_list_gt
print("final perf : ", final_perf)
