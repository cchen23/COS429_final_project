# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 14:26:03 2017

@author: Cathy
"""
from sklearn.datasets import fetch_lfw_people
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

import numpy as np
import time

import algorithms
import manipulations

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

def get_lfw_dataset(min_faces_per_person, manipulation_info):
    """ Return train and test data and labels from 'Labeled Faces in the Wild" dataset."""
    dataset = fetch_lfw_people(min_faces_per_person=min_faces_per_person)
    data = dataset.data # num_people x image_length
    data = manipulations.perform_manipulation(data, manipulation_info)
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

def run_experiment(train_function, predict_function, model_name, manipulation_info, savename):
    """ Trains and tests using train_model_function and evaluate_model_function
    arguments, and saves results. """
    min_faces_per_person = 20
    train_data, train_targets, test_data, test_targets = get_lfw_dataset(min_faces_per_person, manipulation_info)
    
    time1 = time.clock()
    model = train_function(train_data, train_targets)
    time2 = time.clock()
    train_time = time2 - time1

    time1 = time.clock()
    train_predictions = predict_function(model, train_data)
    train_accuracy = compute_accuracy(train_predictions, train_targets)
    test_predictions = predict_function(model, test_data)
    test_accuracy = compute_accuracy(test_predictions, test_targets)
    time2 = time.clock()
    test_time = time2 - time1

    # Print results.
    num_faces = len(np.unique(train_targets))
    print("Manipulation info: %s" % str(manipulation_info))
    print("Recognition Algorithm: %s" % model_name)
    print("Number of distinct faces: %d" % num_faces)
    print("Chance rate: %f" % (1 / num_faces))
    print("Train accuracy: %f" % train_accuracy)
    print("Test accuracy: %f" % test_accuracy)
    print("Training Time: %s sec" % train_time)
    print("Testing Time: %s sec" % test_time)
    print("\n")
    
    # Save results.
    with open("../results/%s_results.txt" % savename, "a") as f:
        f.write("Algorithm: %s\n" % model_name)
        f.write("Min faces per person: %d\n" % min_faces_per_person)
        f.write("Number of distinct faces: %d\n" % num_faces)
        f.write("Chance rate: %f\n" % (1 / num_faces))
        f.write("Train accuracy: %f\n" % train_accuracy)
        f.write("Test accuracy: %f\n" % test_accuracy)
        f.write("Training Time: %s sec\n" % train_time)
        f.write("Testing Time: %s sec\n" % test_time)
        f.write("\n\n")
        
if __name__ == "__main__":
    # Experiments without manipulations.
    manipulation_infos = [("none", -1), ("occlude_lfw", 20), ("occlude_lfw", 10), ("occlude_lfw", 30), ("occlude_lfw", 40), ("radial_distortion", 0.00015), ("radial_distortion", -0.00015), ("radial_distortion", 0.0003), ("radial_distortion", -0.0003), ("radial_distortion", 0.0005), ("radial_distortion", -0.0005)]
    savenames = ["nomanipulation", "occludelfw_20", "occludelfw_10", "occludelfw_30", "occludelfw_40", "radial_distortion_k00015", "radial_distortion_kneg00015", "radial_distortion_k0003", "radial_distortion_kneg0003", "radial_distortion_k0005", "radial_distortion_kneg0005"]
    for (manipulation_info, savename) in zip(manipulation_infos, savenames):
        run_experiment(algorithms.train_pca, algorithms.predict_pca, "PCA", manipulation_info, savename)
        run_experiment(algorithms.train_sparserepresentation, algorithms.predict_sparserepresentation, "Sparse Representation", manipulation_info, savename)
        run_experiment(algorithms.train_sparserepresentation_dimension_reduction, algorithms.predict_sparserepresentation_dimension_reduction, "Sparse Representation Dimension Reduction", manipulation_info, savename)
        run_experiment(algorithms.train_sparserepresentation_combinedl1, algorithms.predict_sparserepresentation_combinedl1, "Sparse Representation Combined l1", manipulation_info, savename)