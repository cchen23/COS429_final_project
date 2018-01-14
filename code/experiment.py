#! /usr/bin/env python3
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
import itertools
import os.path
import csv
import concurrent.futures
import argparse

import algorithms
import manipulations

from manipulations import ManipulationInfo

COLUMNS = [
    "Manipulation Type",
    "Manipulation Parameters",
    "Recognition Algorithm",
    "Min Faces Per Person",
    "Number of Distinct Faces",
    "Chance Rate",
    "Train Accuracy",
    "Test Accuracy",
    "Training Time",
    "Testing Time",
]

def split_traintest(targets, num_train):
    # TODO: Different way of splitting train and test?
    # Maybe should use more than one of each face to train?
    """ Splits targets into train and test indices."""
    unique_targets = np.unique(targets)
    train_indices = []
    test_indices = []

    for label in unique_targets:
        label_indices = np.where(targets == label)[0].tolist()
        train_indices += label_indices[0:num_train]
        test_indices += label_indices[num_train:]

    return train_indices, test_indices

def get_lfw_dataset(min_faces_per_person, num_train):
    """ Return train and test data and labels from 'Labeled Faces in the Wild" dataset."""
    dataset = fetch_lfw_people(min_faces_per_person=min_faces_per_person)
    data = dataset.data # num_people x image_length
    mean_face = np.mean(data, axis=0)
    data = data - mean_face

    train_indices, test_indices = split_traintest(dataset.target, num_train)
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

def run_experiment(model_name, manipulation_info, num_train, savename=None):
    """ Trains and tests using train_model_function and evaluate_model_function
    arguments, and saves results. """
    min_faces_per_person = 20
    train_data, train_targets, test_data_nomanipulation, test_targets = get_lfw_dataset(min_faces_per_person, num_train)

    # Apply manipulations to test dataset
    test_data = np.array(manipulations.perform_manipulation(test_data_nomanipulation, manipulation_info))

    time1 = time.clock()
    model = algorithms.train(model_name, train_data, train_targets)
    time2 = time.clock()
    train_time = time2 - time1

    time1 = time.clock()
    train_predictions = algorithms.predict(model_name, model, train_data)
    train_accuracy = compute_accuracy(train_predictions, train_targets)
    test_predictions = algorithms.predict(model_name, model, test_data)
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
    if savename is not None:
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

    return {
        "Manipulation Type": manipulation_info.type,
        "Manipulation Parameters": manipulation_info.parameters,
        "Recognition Algorithm": model_name,
        "Min Faces Per Person": min_faces_per_person,
        "Number of Distinct Faces": num_faces,
        "Chance Rate": (1 / num_faces),
        "Train Accuracy": train_accuracy,
        "Test Accuracy": test_accuracy,
        "Training Time": train_time,
        "Testing Time": test_time,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Re-run experiments with this model")
    parser.add_argument("--manipulation", help="Re-run experiments with this manipulation")
    parser.add_argument("--num-train", type=int, help="Only run experiments with this number of training images (does not re-run)")
    args = parser.parse_args()

    # Experiments without manipulations.
    model_names = [
        "PCA",
        "Sparse Representation",
        "Sparse Representation Dimension Reduction",
        "Sparse Representation Combined l1",
        "SVM",
    ]
    manipulation_infos = [
        ManipulationInfo("none", {}),
        ManipulationInfo("occlude_lfw", {"occlusion_size": 20}),
        ManipulationInfo("occlude_lfw", {"occlusion_size": 10}),
        ManipulationInfo("occlude_lfw", {"occlusion_size": 30}),
        ManipulationInfo("occlude_lfw", {"occlusion_size": 40}),
        ManipulationInfo("radial_distortion", {"k": 0.00015}),
        ManipulationInfo("radial_distortion", {"k": -0.00015}),
        ManipulationInfo("radial_distortion", {"k": 0.0003}),
        ManipulationInfo("radial_distortion", {"k": -0.0003}),
        ManipulationInfo("radial_distortion", {"k": 0.0005}),
        ManipulationInfo("radial_distortion", {"k": -0.0005}),
        ManipulationInfo("blur", {"blurwindow_size": 5}),
        ManipulationInfo("blur", {"blurwindow_size": 10}),
    ]
    num_trains = [10, 15, 19]

    should_rerun = False
    if args.model:
        should_rerun = True
        if args.model not in model_names:
            raise Exception("Unrecognized model name")
        model_names = [model for model in model_names if model == args.model]
    if args.manipulation:
        should_rerun = True
        manipulation_infos = [mi for mi in manipulation_infos if mi.type == args.manipulation]
        if len(manipulation_infos) == 0:
            raise Exception("Unrecognized manipulation type")
    if args.num_train:
        num_trains = [args.num_train]

    for num_train in num_trains:
        print("Num training examples: %d" % num_train)
        save_path = "../results/results_%d.csv" % num_train

        # Create new save file if it doesn't exist
        if not os.path.exists(save_path):
            with open(save_path, 'w') as f:
                csv.DictWriter(f, fieldnames=COLUMNS).writeheader()

        # Load existing results
        with open(save_path, 'r') as f:
            seen_results = [(row["Manipulation Type"], row["Manipulation Parameters"], row["Recognition Algorithm"]) for row in csv.DictReader(f)]

        # Run experiments
        with open(save_path, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)

            for manipulation_info in manipulation_infos:
                for model_name in model_names:
                    params_tuple = (manipulation_info.type, str(manipulation_info.parameters), model_name)

                    # Skip completed experiments
                    if params_tuple in seen_results and not should_rerun:
                        print("Skipping: %s" % str(params_tuple))
                        continue

                    print("Running: %s" % str(params_tuple))
                    try:
                        results = run_experiment(model_name, manipulation_info, num_train)
                        writer.writerow(results)
                        f.flush()
                    except Exception as e:
                        print("Error:" + e)
