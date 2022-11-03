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


def create_data(trainset, shot_number=1, num_class=10):
    """
    create a few_shot data
    """
    data_fewshot = DataFewShot(num_class)
    number_sample = np.zeros(num_class)
    total_sample = len(trainset)
    iteration = 0

    while not (np.all(number_sample == shot_number)) and iteration < total_sample:
        img, classe = trainset[iteration]
        if number_sample[classe] < shot_number:
            print(f"adding one sample to class {classe}")
            representation = few_shot_model.get_features(img)
            data_fewshot.add_repr(classe, representation)
            data_fewshot.add_mean_repr(representation, DEVICE)
            number_sample[classe] += 1
        iteration += 1
    data_fewshot.aggregate_mean_rep()
    return data_fewshot


def get_performance(dataset, data_fewshot):
    correct_pred = 0
    incorect_pred = 0

    list_pred = []
    list_gt = []
    for i, (img, gt_classe) in enumerate(dataset):

        if i == 1000:
            break

        classe_pred, prediction = few_shot_model.predict_class(
            img, data_fewshot.shot_list, data_fewshot.mean_features
        )

        list_pred.append(classe_pred)
        list_gt.append(gt_classe)

        if classe_pred == gt_classe:
            correct_pred += 1
        else:
            incorect_pred += 1
    return correct_pred / (correct_pred + incorect_pred), list_pred, list_gt


TRAIN, TEST, CLASSES = get_loader()


BACKBONE_SPECS = {
    "model_name": "resnet12",
    "path": "weight/tieredlong1.pt1",
    "kwargs": {
        "feature_maps": 64,
        "input_shape": [3, 84, 84],
        "num_classes": 351,  # 64
        "few_shot": True,
        "rotations": False,
    },
}


# model parameters
CLASSIFIER_SPECS = {"model_name": "knn", "kwargs": {"number_neighboors": 1}}
DEVICE = "cuda:0"
TRANSFORMS = get_camera_preprocess()
# TRANSFORMS = transforms.Compose(
#         [transforms.ToTensor(),
#         transforms.Normalize(
#             [0.5,0.5,0.5],# np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
#             [0.5,0.5,0.5]# np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]),
#         ),
#         transforms.Resize(1)])

# number to save
NUMBER_SHOT = 6


few_shot_model = FewShotModel(BACKBONE_SPECS, CLASSIFIER_SPECS, TRANSFORMS, DEVICE)
# model related

data = create_data(TRAIN, shot_number=NUMBER_SHOT)

final_perf, _,_ = get_performance(TEST, data)#final_list_pred, final_list_gt
print(final_perf)
