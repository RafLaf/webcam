"""
neural network modules
handle loading, inference and prediction
"""

import numpy as np


from utils import softmax, one_hot, k_small


def feature_preprocess(features, mean_base_features):
    """

    preprocess the feature (normalisation on the unit sphere) for classification
        Args :
            features(np.ndarray) : feature to be preprocessed
            mean_base_features(np.ndarray) : expected mean of the tensor
        returns:
            features(np.ndarray) : normalized feature
    """
    features = features - mean_base_features
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    return features


class FewShotModel:
    """
    class defining a few shot model
        attributes :
            - backbone : initialized with backbone_specs(dict):
                specs defining the how to load the backbone
            - classifier_specs :
                parameters of the final classification model
            - preprocess : how preprocess input image
            - device : on wich device should the computation take place
    """

    def __init__(self, classifier_specs):

        self.classifier_specs = classifier_specs

    def predict_class_feature(self, features, recorded_data, preprocess_feature=True):
        """
        predict the class of a features with a model

        args:
            img(PIL Image or numpy.ndarray) : current img that we will predict
            recorded_data (DataFewShot) : data used for classification
            model_name : wich model do we use
            **kwargs : additional parameters of the model
        returns :
            classe_prediction : class prediction
            probas : probability of belonging to each class
        """

        mean_feature = recorded_data.get_mean_features()

        model_name = self.classifier_specs["model_name"]
        model_arguments = self.classifier_specs["kwargs"]
        shots_list = recorded_data.get_shot_list()

        if preprocess_feature:

            features = feature_preprocess(features, mean_feature)

        # class asignement using the corespounding model

        if model_name == "ncm":

            shots = np.stack([np.mean(shot, axis=0) for shot in shots_list], axis=0)
            # shots=shots.detach().cpu().numpy()
            if preprocess_feature:
                shots = feature_preprocess(shots, mean_feature)

            distances = np.linalg.norm(shots - features, axis=1, ord=2)

            probas = softmax(-20 * distances, dim=0)

        elif model_name == "knn":
            number_neighboors = model_arguments["number_neighboors"]
            # create target list of the shots

            shots = np.concatenate(shots_list)

            if preprocess_feature:
                shots = feature_preprocess(shots, mean_feature)

            targets = np.concatenate(
                [
                    np.array(i * np.ones(shots_list[i].shape[0], dtype=np.int64))
                    for i in range(len(shots_list))
                ],
                axis=0,
            )

            # shots=shots
            # features=features.detach().cpu().numpy()

            distances = np.linalg.norm(shots - features, axis=1, ord=2)
            # distances=(shots - features)@(shots - features)
            # #L2 because unit circle
            # get the k nearest neighbors

            indices = k_small(distances, number_neighboors)

            probas = one_hot(targets[indices], len(shots_list))
            probas = np.sum(probas, axis=0) / number_neighboors

        else:
            raise NotImplementedError(f"classifier : {model_name} is not implemented")

        classe_prediction = probas.argmax()
        return classe_prediction, probas

    def predict_class_moving_avg(self, features, prev_probabilities, recorded_data):
        """

        update the probabily and attribution of having a class, using the current image
        args :
            img(PIL Image or numpy.ndarray) : current img,
            prev_probabilities(?) : probability of each class for previous prediction
            recorded_data (DataFewShot) : data recorded for classification

        returns :
            classe_prediction : class prediction
            probas : probability of belonging to each class
        """
        model_name = self.classifier_specs["model_name"]

        _, current_proba = self.predict_class_feature(features, recorded_data)

        print("probabilities:", current_proba)

        if prev_probabilities is None:
            probabilities = current_proba
        else:
            if model_name == "ncm":
                probabilities = prev_probabilities * 0.85 + current_proba * 0.15
            elif model_name == "knn":
                probabilities = prev_probabilities * 0.95 + current_proba * 0.05

        classe_prediction = probabilities.argmax()
        return classe_prediction, probabilities
