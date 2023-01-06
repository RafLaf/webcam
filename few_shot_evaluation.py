import numpy as np


#from memory_profiler import profile

from args import args
from few_shot_model.few_shot_eval import get_features_numpy
from few_shot_model.dataset_numpy import get_dataset_numpy
from torch_evaluation.backbone_loader import get_model
from few_shot_model.few_shot_eval import define_runs#,get_features_few_shot_ds,
from few_shot_model.few_shot_model import FewShotModel

#@profile#comment/uncoment and flag -m memory_profiler after python
def launch_program(BACKBONE_SPECS):

  
    #from lim_ram import set_limit
    backbone = get_model(BACKBONE_SPECS)

    #set_limit(6000*1024*1024)
     #import torch_evaluation.datasets as datasets
    # loaders, input_shape, num_classes, few_shot, top_5 = datasets.get_dataset(args.dataset)


    # if few_shot:
    #     train_loader, train_clean, val_loader, test_loader = loaders
    #     num_classes, val_classes, novel_classes, elements_per_class = num_classes
    # else:
    #     train_loader, val_loader, test_loader = loaders
        
    #data=get_dataset_numpy("data/cifar-10-batches-py/test_batch")
    data=get_dataset_numpy(args.dataset_path)
    
    #features=get_features_few_shot_ds(backbone,test_loader,n_aug=args.sample_aug)
    data=(data/255-np.array([0.485, 0.456, 0.406],dtype=data.dtype))/ np.array([0.229, 0.224, 0.225],dtype=data.dtype)
    features=get_features_numpy(backbone,data)
 
    #features=torch.load("weight/cifarfeatures1.    pt11",map_location="cpu").cpu().numpy()[80:]


    sample_per_class=features.shape[1]
    nshots=5
    num_classes=10
    #sample_per_class=600
    classe,index=define_runs(args.n_runs,args.n_ways, nshots, args.n_queries, num_classes, [sample_per_class]*num_classes) 
    #cifar10 : 122mb
    # runs : 84kb

    index_shots,index_queries = index[:,:,:nshots],index[:,:,nshots:]

    #TODO : adapt this in order to put it inside the loop
    
    extracted_shots=features[np.stack([classe]*nshots,axis=-1),index_shots]#compute features corresponding to each experiment
    extracted_queries=features[np.stack([classe]*args.n_queries,axis=-1),index_queries]#compute features corresponding to each experiment
    # settings : 5-ways,15 queries, 10 classes, 1000 runs
    #extracted_shots : 45mb
    #extracted queries : 134mb
    mean_feature=np.mean(extracted_shots,axis=(1,2))
    #mean features : 9mb

    bs=15
    
    CLASSIFIER_SPECS = args.classifier_specs
    fs_model=FewShotModel(CLASSIFIER_SPECS)
    perf=[]
    for i in range(args.n_runs//bs):
        #view, no data
        batch_q=extracted_queries[i*bs:(i+1)*bs]
        

        batch_shot=extracted_shots[i*bs:(i+1)*bs]
        batch_mean_feature=mean_feature[i*bs:(i+1)*bs]

        predicted_class,_=fs_model.predict_class_batch(batch_q,batch_shot,batch_mean_feature)
        perf.append(np.mean(predicted_class==np.expand_dims(np.arange(0,5),axis=(0,2))))
    print("perf : ",np.mean(perf)," +-",np.std(perf))
    print("dn")

    


# BACKBONE_SPECS={
#     "type":"pytorch_batch",
#     "device":"cuda:0",
#     "model_name": "resnet12",  
#     "kwargs": {
#         "input_shape": [3, 32, 32],
#         "num_classes": 64,  # 351,
#         "few_shot": True,
#         "rotations": False,
#     },

# }

BACKBONE_SPECS = args.backbone_specs

# tieredlong1.pt1",
# if args.backbone_type=="cifar_small":
#     BACKBONE_SPECS["path"]="weight/smallcifar1.pt1"
#     BACKBONE_SPECS["kwargs"]["feature_maps"]=45

# elif args.backbone_type=="cifar":
#     BACKBONE_SPECS["path"]="weight/cifar1.pt1"
#     BACKBONE_SPECS["kwargs"]["feature_maps"]=64

# elif args.backbone_type=="cifar_tiny":
#     BACKBONE_SPECS["path"]="weight/tinycifar1.pt1"
#     BACKBONE_SPECS["kwargs"]["feature_maps"]=32
#     print(BACKBONE_SPECS)


if __name__=="__main__":
    #set_limit(500*1024*1024)#simulate the memmory limitation of the pynk
    launch_program(BACKBONE_SPECS)