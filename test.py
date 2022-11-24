
from backbone import get_model#get_camera_preprocess
from few_shot_eval import get_features,define_runs
import datasets
from args import args
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
DEVICE = "cuda:0"


backbone = get_model(BACKBONE_SPECS, DEVICE,use_batch=True)
loaders, input_shape, num_classes, few_shot, top_5 = datasets.get_dataset(args.dataset)
if few_shot:
    train_loader, train_clean, val_loader, novel_loader = loaders
    num_classes, val_classes, novel_classes, elements_per_class = num_classes
else:
    train_loader, val_loader, test_loader = loaders
    
features=get_features(backbone,test_loader,n_aug=args.sample_aug)
runs=[define_runs(n_runs,n_ways, s, n_queries, num_classes, elements_per_class) for s in n_shots_list]

print("dn")

