# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 14:26:03 2017

@author: Cathy
"""
from sklearn.datasets import fetch_lfw_people
#from sklearn.model_selection import train_test_split

import algorithms
import numpy as np

from sklearn.preprocessing import normalize

def split_traintest(targets):
    # TODO: Different way of splitting train and test?
    # Maybe should use more than one of each face to train?
    """ Splits targets into train and test indices."""
    unique_targets = np.unique(targets)
    num_train = 3
    train_indices = []
    test_indices = []

    for label in unique_targets:
        label_indices = np.where(targets == label)[0].tolist()
        train_indices += label_indices[0:num_train]
        test_indices += label_indices[num_train:]

    return train_indices, test_indices

def get_lfw_dataset(min_faces_per_person):
    """ Return train and test data and labels from 'Labeled Faces in the Wild" dataset."""
    dataset = fetch_lfw_people(min_faces_per_person=min_faces_per_person)
    data = dataset.data # num_people x image_length
    mean_face = np.mean(data, axis=0)
    data = data - mean_face

    train_indices, test_indices = split_traintest(dataset.target)
    train_data = data[train_indices,:]
    train_targets = dataset.target[train_indices]
    test_data = data[test_indices,:]
    test_targets = dataset.target[test_indices]

    test_data = normalize(test_data, axis=1)
    train_data = normalize(train_data, axis=1)

    #train_data, test_data, train_targets, test_targets = train_test_split(data, dataset.target)
    return train_data, train_targets, test_data, test_targets

def compute_accuracy(predictions, targets):
    accuracy = np.sum(np.array(predictions) == np.array(targets)) / len(predictions)
    return accuracy

def run_experiment(train_function, predict_function, model_name):
    """ Trains and tests using train_model_function and evaluate_model_function
    arguments, and saves results. """
    min_faces_per_person = 20
    train_data, train_targets, test_data, test_targets = get_lfw_dataset(min_faces_per_person)
    
    model = train_function(train_data, train_targets)
    train_predictions = predict_function(model, train_data)
    train_accuracy = compute_accuracy(train_predictions, train_targets)
    test_predictions = predict_function(model, test_data)
    test_accuracy = compute_accuracy(test_predictions, test_targets)
    
    # Print results.
    num_faces = len(np.unique(train_targets))
    print("Recognition Algorithm: %s" % model_name)
    print("Number of distinct faces: %d" % num_faces)
    print("Chance rate: %f" % (1 / num_faces))
    print("Train accuracy: %f" % train_accuracy)
    print("Test accuracy: %f" % test_accuracy)
    print("\n")
    
    # Save results.
    with open("../results/%s_results.txt" % (model_name).replace(" ",""), "a") as f:
        f.write("Min faces per person: %d\n" % min_faces_per_person)
        f.write("Number of distinct faces: %d\n" % num_faces)
        f.write("Chance rate: %f\n" % (1 / num_faces))
        f.write("Train accuracy: %f\n" % train_accuracy)
        f.write("Test accuracy: %f\n" % test_accuracy)
        f.write("\n\n")
        
if __name__ == "__main__":
    run_experiment(algorithms.train_pca, algorithms.predict_pca, "PCA")
    run_experiment(algorithms.train_sparserepresentation, algorithms.predict_sparserepresentation, "Sparse Representation")
    run_experiment(algorithms.train_sparserepresentation_dimension_reduction, algorithms.predict_sparserepresentation_dimension_reduction, "Sparse Representation Dimension Reduction")
    run_experiment(algorithms.train_sparserepresentation_combinedl1, algorithms.predict_sparserepresentation_combinedl1, "Sparse Representation Combined l1")