# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 14:26:03 2017

@author: Cathy
"""
from sklearn.datasets import fetch_lfw_people

import pickle as p
import experiment_utils as utils
import numpy as np

def split_traintest(target):
    # TODO: Different way of splitting train and test?
    # Maybe should use more than one of each face to train?
    """ Splits targets into train and test indices."""
    unique_targets = np.unique(target)
    num_train = 1
    train_indices = []
    test_indices = []

    for label in unique_targets:
        label_indices = np.where(target == label)[0].tolist()
        train_indices += label_indices[0:num_train]
        test_indices += label_indices[num_train:]

    return train_indices, test_indices

def get_lfw_dataset():
    """ Return train and test data and labels from 'Labeled Faces in the Wild" dataset."""
    min_faces_per_person = 5
    dataset = fetch_lfw_people(min_faces_per_person=min_faces_per_person)
    data = dataset.data
    mean_face = np.mean(data, axis=0)
    data = data - mean_face
    
    train_indices, test_indices = split_traintest(dataset.target)
    train_data = data[train_indices,:]
    train_target = dataset.target[train_indices]
    test_data = data[test_indices,:]
    test_target = dataset.target[test_indices]

    return train_data, train_target, test_data, test_target

def run_experiment(train_model_function, evaluate_model_function, model_name):
    """ Trains and tests using train_model_function and evaluate_model_function
    arguments, and saves results. """
    train_data, train_target, test_data, test_target = get_lfw_dataset()
    
    model = train_model_function(train_data, train_target)
    return model
    # results = evaluate_model_function(model, test_data, test_target)
    #p.dump(results, open( "../results/%s_results.p" % model_name, "wb" ))

if __name__ == "__main__":
    model = run_experiment(utils.train_pca, utils.evaluate_pca, "PCA")
    weights, train_target = model
    print(weights.shape)
    print(train_target)