"""
manage saved data for the few shot algorithm
"""
import torch

class DataFewShot:
    """represent the data saved for few shot learning
    attributes :
        num_classe : max number of class handled
        mean_features(torch.Tensor or list(torch.Tensor)) : mean of the feature / list of feature to aggregate 
        registered_classes : registered class
        shot_list : list of the regitered data
    """

    def __init__(self,num_class):
        self.num_class=num_class
        self.shot_list=list(range(num_class))
        self.mean_features=[]
        self.registered_classes=[]
        self.mean_repr=[]

    def add_repr(self,classe,repr):
        """
        add the given repr to the given classe
        """
        if classe not in self.registered_classes:
            self.registered_classes.append(classe)
            self.shot_list[classe]=repr
        else:
            #TODO : change dtype to numpy array
            self.shot_list[classe] = torch.cat(
                (self.shot_list[classe], repr), dim=0
            )

    

    def aggregate_mean_rep(self):
        """
        aggregate all saved features
        can only be called once

        """
        self.mean_features = torch.cat(self.mean_features, dim=0)
        self.mean_features = self.mean_features.mean(dim=0)
        
    def add_mean_repr(self,features,device):
        """
        add a given image to the mean repr of the datas
        """
        
        self.mean_features.append(features.detach().to(device))

    
    def reset(self):
        """
        reset the saved image, but not the mean repr
        """
        self.shot_list=list(range(self.num_class))
        self.registered_classes=[]