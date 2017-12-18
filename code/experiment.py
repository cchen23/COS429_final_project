# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 14:26:03 2017

@author: Cathy
"""
from sklearn.datasets import fetch_lfw_people

import pickle as p
#import experiment_utils as utils
import numpy as np

def split_traintest(target):
    unique_targets = np.unique(target)
    train_indices = []
    test_indices = []
    
    for label in unique_targets:
        label_indices = np.where(target == label)[0].tolist()
        train_indices += [label_indices[0]]
        test_indices += label_indices[1:]

    return train_indices, test_indices

def get_lfw_dataset():
    """ Return train and test data and labels from 'Labeled Faces in the Wild" dataset."""
    min_faces_per_person = 5
    dataset = fetch_lfw_people(min_faces_per_person=min_faces_per_person)
    
    train_indices, test_indices = split_traintest(dataset.target)
    train_data = dataset.data[train_indices,:]
    train_target = dataset.target[train_indices]
    test_data = dataset.data[test_indices,:]
    test_target = dataset.target[test_indices]
    
    return train_data, train_target, test_data, test_target

def run_experiment(train_model_function, evaluate_model_function, model_name):
    train_data, train_target, test_data, test_target = get_lfw_dataset()
    
    model = train_model_function(train_data, train_target)
    results = evaluate_model_function(model, test_data, test_target)
    
    p.dump(results, open( "%s_results.p" % model_name, "wb" ))

if __name__ == "__main__":
    train_data, train_target, test_data, test_target = get_lfw_dataset()
    print(train_data.shape)
    print(train_target.shape)
    print(test_data.shape)
    print(test_target.shape)