"""
neural network modules
handle loading, inference and prediction
"""
# on veut que le truc soit capable de gérer des trucs de la forme:

import numpy as np


from few_shot_model.numpy_utils import softmax, one_hot, k_small


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
    features = features / np.linalg.norm(features, axis=-1, keepdims=True)
    return features


def ncm(shots_mean, features):
    """
    compute the class attribution probas using the ncm classifier
    args :
        - shots_mean array(...,n_class,n_dim) : mean of the saved shots for each classe
        - features array(...,n_dim) : features to classify (leading dims same as previous array)
    """
    features = np.expand_dims(features, axis=-2)  # broadcastable along class axis
    distances = np.linalg.norm(shots_mean - features, axis=-1, ord=2)
    probas = softmax(-20 * distances, dim=-1)
    return probas


def knn(shots_points, features, target, number_neighboors):
    """
    compute the class attribution probas using the ncm classifier
    args :
        - shots_mean array(...,n_points,n_dim) : mean of the saved shots for each classe
        - features array(...,n_dim) : features to classify (leading dims same as previous array)
        - target : array(n_points) : represent feature assignement. Expected to have value in [0, ...,n_class-1]
        - number_neighboors (int) : number of neighboors to take
    """
    number_class = np.max(target) + 1

    features = np.expand_dims(features, axis=-2)  # broadcastable along point axis
    distances = np.linalg.norm(shots_points - features, axis=-1, ord=2)
    # distances=(shots - features)@(shots - features)
    # #L2 because unit circle
    # get the k nearest neighbors

    indices = k_small(distances, number_neighboors, axis=-1)  # smalest along exemples

    probas = one_hot(target[indices], number_class)
    probas = (
        np.sum(probas, axis=-2) / number_neighboors
    )  # sum along k neirest neighboors

    return probas


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

    2 possible types of prediction (batchified or not)
    """

    def __init__(self, classifier_specs):
        self.classifier_specs = classifier_specs

    def predict_class_batch(
        self, features, shot_array, mean_feature, preprocess_feature=True
    ):
        """
        predict the class of a features
        args:
            features :
                - (np.ndarray(n_batch,nways,n_queries,n_features)) : features of the current img
            shot_array :
                - array(n_batch,n_ways,n_shots,n_features) (each element of sequence = 1 class)
            mean_feature :
                - array(n_batch,n_features)
            model_name : wich model do we use
            **kwargs : additional parameters of the model
        returns :
            classe_prediction : class prediction
            probas (1,n_features) : probability of belonging to each class

        """
        model_name = self.classifier_specs["model_name"]
        model_arguments = self.classifier_specs.get("kwargs", {})
        # shots_list = recorded_data.get_shot_list()

        if preprocess_feature:
            # (n_batch,1,1,n_features)
            features = feature_preprocess(
                features, np.expand_dims(mean_feature, axis=(1, 2))
            )

        # class asignement using the correspounding model

        if model_name == "ncm":
            shots = np.mean(shot_array, axis=2)  # mean of the shots
            # (n_batch,n_ways,n_features)
            # shots=shots.detach().cpu().numpy()
            if preprocess_feature:
                # (n_batch,1,n_features)
                shots = feature_preprocess(shots, np.expand_dims(mean_feature, axis=1))
            shots = np.expand_dims(shots, axis=(1, 2))
            # (_batch,1,1,n_ways,n_features)
            probas = ncm(shots, features)

        elif model_name == "knn":
            number_neighboors = model_arguments["number_neighboors"]
            # create target list of the shots
            n_ways = shot_array.shape[1]
            n_shots = shot_array.shape[2]
            shots = np.reshape(
                shot_array,
                axis=(shot_array.shape[0], n_ways * n_shots, shot_array.shape[3]),
            )
            # shots : (n_batch,n_exemples,nfeatures)
            if preprocess_feature:
                shots = feature_preprocess(shots, np.expand_dims(mean_feature, axis=1))
            shots = np.expand_dims(shots, axis=(2, 3))
            # (_batch,n_ways,1,n_features)

            targets = np.concatenate(
                [
                    class_id * np.ones(n_shots, dtype=np.int64)
                    for class_id in range(n_ways)
                ],
                axis=0,
            )

            probas = knn(shots, features, targets, number_neighboors)

        else:
            raise NotImplementedError(f"classifier : {model_name} is not implemented")

        classe_prediction = np.argmax(probas, axis=-1)
        return classe_prediction, probas

    def predict_class_feature(
        self, features, shots_list, mean_feature, preprocess_feature=True
    ):
        """
        predict the class of a features

        args:
            features :
                - (np.ndarray(n_features)) : features of the current img
            shot_list :
                - sequence(array(n_shots_i,n_features)) (each element of sequence = 1 class)
            mean_feature :
                - array(n_features)
            model_name : wich model do we use
            **kwargs : additional parameters of the model
        returns :
            classe_prediction : class prediction
            probas (1,n_features) : probability of belonging to each class
        """

        # mean_feature = np.mean(shots_list,axis=0) #recorded_data.get_mean_features()

        model_name = self.classifier_specs["model_name"]
        model_arguments = self.classifier_specs.get("kwargs", {})
        # shots_list = recorded_data.get_shot_list()

        if preprocess_feature:
            features = feature_preprocess(features, mean_feature)

        # class asignement using the correspounding model

        if model_name == "ncm":
            shots = np.stack(
                [np.mean(shot, axis=0) for shot in shots_list], axis=0
            )  # sequence -> array
            # shots : (nclass,nfeatures)
            # shots=shots.detach().cpu().numpy()
            if preprocess_feature:
                shots = feature_preprocess(shots, mean_feature)
            probas = ncm(shots, features)

        elif model_name == "knn":
            number_neighboors = model_arguments["number_neighboors"]
            # create target list of the shots

            shots = np.concatenate(shots_list, axis=0)  # sequence -> array
            # shots : (n_exemples,nfeatures)

            if preprocess_feature:
                shots = feature_preprocess(shots, mean_feature)

            number_ways = len(shots_list)  # ok for sequence and array

            targets = np.concatenate(
                [
                    class_id * np.ones(shots_list[class_id].shape[0], dtype=np.int64)
                    for class_id in range(number_ways)
                ],
                axis=0,
            )

            probas = knn(shots, features, targets, number_neighboors)

        else:
            raise NotImplementedError(f"classifier : {model_name} is not implemented")

        classe_prediction = np.argmax(probas, axis=-1)
        return classe_prediction, probas

    def predict_class_moving_avg(
        self, features, prev_probabilities, shots_list, mean_feature
    ):
        """

        update the probabily and attribution of having a class, using the current image
        args :
            features(np.ndarray((1,n_features))) : features of the current img
            prev_probabilities(?) : probability of each class for previous prediction
            recorded_data (DataFewShot) : data recorded for classification

        returns :
            classe_prediction : class prediction
            probas : probability of belonging to each class
        """
        model_name = self.classifier_specs["model_name"]

        _, current_proba = self.predict_class_feature(
            features, shots_list, mean_feature
        )

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
