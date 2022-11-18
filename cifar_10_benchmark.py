"""
test the performance of the demo using the cifar10 dataset
"""

import torch
import torchvision
from torchvision import transforms
import numpy as np

from few_shot_model import FewShotModel, get_camera_preprocess
from data_few_shot import DataFewShot


def get_loader():
    """
    get the loader, see :
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

    """

    # batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)

    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                        shuffle=False, num_workers=1)

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    return trainset, testset, classes


def create_data(trainset, shot_number=1, num_class=10,mean_computing_type="SNAPSHOT",data_aug=None):
    """
    create a few_shot data
    """
    data_fewshot = DataFewShot(num_class)
    number_sample = np.zeros(num_class,dtype=np.int)
    total_sample = len(trainset)
    iteration = 0
    print("total sample in training :", total_sample)

    while  (iteration < total_sample):#and (np.any(number_sample < shot_number))
        img, classe = trainset[iteration]
        if number_sample[classe] < shot_number:
            
            representation = few_shot_model.get_features(img,augmentation=data_aug)
            data_fewshot.add_repr(classe, representation)
            
            number_sample[classe] += 1
            if mean_computing_type=="SNAPSHOT":
                data_fewshot.add_mean_repr(representation)
        if mean_computing_type=="ALL":
            data_fewshot.add_mean_repr(representation)

        iteration += 1
    print("number of image for computing mean : ",len(data_fewshot.mean_features))
    print("number of sample per class",number_sample)
    data_fewshot.aggregate_mean_rep(DEVICE)
    #print("mean repr",data_fewshot.mean_features)
    return data_fewshot


def get_performance(dataset, data_fewshot):
    correct_pred = 0
    incorect_pred = 0

    list_pred = []
    list_gt = []
    for i, (img, gt_classe) in enumerate(dataset):

        if i == 1000:
            break

        classe_pred, prediction = few_shot_model.predict_class(img, data_fewshot)

        list_pred.append(classe_pred)
        list_gt.append(gt_classe)

        if classe_pred == gt_classe:
            correct_pred += 1
        else:
            incorect_pred += 1

    return correct_pred / (correct_pred + incorect_pred), list_pred, list_gt

torch.manual_seed(0)
TRAIN, TEST, CLASSES = get_loader()





# model parameters


BACKBONE_SPECS = {
    "model_name": "resnet12",
    "path": "weight/cifar1.pt1",#tieredlong1.pt1",
    "kwargs": {
        "feature_maps": 64,
        "input_shape": [3, 32, 32],
        "num_classes": 64,#351,
        "few_shot": True,
        "rotations": False,
    },
}

CLASSIFIER_SPECS = {
    "model_name": "ncm", 
    "kwargs":{}# {"number_neighboors": 1}
}
DEVICE = "cuda:0"

#TRANSFORMS = get_camera_preprocess()

TRANSFORMS = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)# np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]),
        ),
        #transforms.Resize(42),
        #transforms.CenterCrop(32)
        ])

# number to save
NUMBER_SHOT = 5


few_shot_model = FewShotModel(BACKBONE_SPECS, CLASSIFIER_SPECS, TRANSFORMS, DEVICE)
# model related

sub_train=torch.utils.data.Subset(TRAIN,[i for i in range(100)])

data = create_data(TRAIN, shot_number=NUMBER_SHOT,mean_computing_type="SNAPSHOT")
final_perf, a,b = get_performance(TEST, data)#final_list_pred, final_list_gt
print(final_perf)