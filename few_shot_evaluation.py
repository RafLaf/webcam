import numpy as np


#from memory_profiler import profile

from args import args
from performance_evaluation.few_shot_eval import get_features_numpy
from performance_evaluation.dataset_numpy import get_dataset_numpy
from backbone_loader.backbone_loader import get_model
from performance_evaluation.few_shot_eval import define_runs#,get_features_few_shot_ds,
from few_shot_model.few_shot_model import FewShotModel

#@profile#comment/uncoment and flag -m memory_profiler after python
def launch_program(BACKBONE_SPECS):

    #from lim_ram import set_limit
    backbone = get_model(BACKBONE_SPECS)
    data=get_dataset_numpy(args.dataset_path)
    
    data=(data/255-np.array([0.485, 0.456, 0.406],dtype=data.dtype))/ np.array([0.229, 0.224, 0.225],dtype=data.dtype)
    
    features=get_features_numpy(backbone,data,args.batch_size)


    sample_per_class=features.shape[1]
    nshots=args.n_shots
    num_classes=args.num_classes_dataset
    #sample_per_class=600
    classe,index=define_runs(args.n_runs,args.n_ways, nshots, args.n_queries, num_classes, [sample_per_class]*num_classes) 
    #cifar10 : 122mb
    # runs : 84kb

    index_shots,index_queries = index[:,:,:nshots],index[:,:,nshots:]
    extracted_shots=features[np.stack([classe]*nshots,axis=-1),index_shots]#compute features corresponding to each experiment
    extracted_queries=features[np.stack([classe]*args.n_queries,axis=-1),index_queries]#compute features corresponding to each experiment

    mean_feature=np.mean(extracted_shots,axis=(1,2))

    bs=args.batch_size_fs
    
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

    


BACKBONE_SPECS = args.backbone_specs


if __name__=="__main__":
    #set_limit(500*1024*1024)#simulate the memmory limitation of the pynk
    launch_program(BACKBONE_SPECS)