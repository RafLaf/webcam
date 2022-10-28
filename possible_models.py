import torch

from resnet12 import ResNet12


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

def get_model(model,model_specs):
    if model=="resnet12":
        return ResNet12(**model_specs)
    else:
        raise NotImplementedError(f"model {model} is not implemented")

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