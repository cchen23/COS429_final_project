#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 14:26:03 2017

@author: Cathy
"""
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets.lfw import check_fetch_lfw
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from scipy.misc import imread, imresize

import numpy as np
import time
import itertools
import os
import os.path
import csv
import concurrent.futures
import argparse
import pandas as pd
import subprocess

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

def get_lfw_dataset(min_faces_per_person, num_train, color=False, size=50):
    """ Return train and test data and labels from 'Labeled Faces in the Wild" dataset."""
    train_data, train_targets, test_data, test_targets = [], [], [], []
    person_index = 0

    for person in os.listdir(os.path.join("images", "lfw_aegan")):
        if not os.path.isfile(get_lfw_image_path(person, min_faces_per_person)):
            continue

        # Load train data
        train_data += [get_lfw_image(get_lfw_image_path(person, index + 1), scale=size/100, color=color) for index in range(min_faces_per_person - num_train, min_faces_per_person)]
        train_targets += [person_index] * num_train
        assert(len(train_data) == len(train_targets))

        # Load test data
        test_data += [get_lfw_image(get_lfw_image_path(person, index + 1), scale=size/100, color=color) for index in range(0, min_faces_per_person - num_train)]
        test_targets += [person_index] * (min_faces_per_person - num_train)
        assert(len(test_data) == len(test_targets))

        person_index += 1

    assert(len(train_data) > 0)
    assert(len(test_data) > 0)

    train_data = np.array(train_data)
    test_data = np.array(test_data)

    mean_face = np.mean(train_data, axis=0)
    train_data -= mean_face
    test_data -= mean_face

    train_data = normalize(train_data.reshape(train_data.shape[0], -1), axis=1).reshape(train_data.shape)
    test_data = normalize(test_data.reshape(test_data.shape[0], -1), axis=1).reshape(test_data.shape)

    return train_data, train_targets, test_data, test_targets

def get_lfw_image_path(person, imagenum):
    return os.path.join("images", "lfw_aegan", person, "{}_{:04d}.jpg".format(person, imagenum))

def get_lfw_dfi_image_path(person, imagenum, transform):
    return os.path.join("images", "dfi", "{}_{:04d}__{}.jpg".format(person, imagenum, transform))

def get_lfw_image(image_path, scale, color=False):
    mode = "RGB" if color else "L"
    face = imread(image_path, mode=mode)
    if scale != 1:
        face = imresize(face, scale)
    if not color:
        # TODO: should be a separate parameter
        face = face.flatten()
    face = face / 255 # convert to float
    return face

def get_lfw_dfi_dataset(min_faces_per_person, num_train, manipulation_info, color=False, size=50):
    assert(manipulation_info.type == "dfi")

    train_data, train_targets, test_data, test_targets = [], [], [], []
    transform = manipulation_info.parameters["transform"]
    person_index = 0

    for person in os.listdir(os.path.join("images", "lfw_aegan")):
        if not os.path.isfile(get_lfw_image_path(person, min_faces_per_person)):
            continue

        # Load train data
        train_data += [get_lfw_image(get_lfw_image_path(person, index + 1), scale=size/100, color=color) for index in range(min_faces_per_person - num_train, min_faces_per_person)]
        train_targets += [person_index] * num_train
        assert(len(train_data) == len(train_targets))

        # Load test data
        person_image_paths = [get_lfw_dfi_image_path(person, index + 1, transform) for index in range(0, min_faces_per_person - num_train)]
        person_image_paths = [image_path for image_path in person_image_paths if os.path.isfile(image_path)]
        assert(1 <= len(person_image_paths) <= min_faces_per_person - num_train)
        test_data += [get_lfw_image(image_path, scale=size/200, color=color) for image_path in person_image_paths]
        test_targets += [person_index] * len(person_image_paths)
        assert(len(test_data) == len(test_targets))

        person_index += 1

    assert(len(train_data) > 0)
    assert(len(test_data) > 0)

    train_data = np.array(train_data)
    test_data = np.array(test_data)

    mean_face = np.mean(train_data, axis=0)
    train_data -= mean_face
    test_data -= mean_face

    train_data = normalize(train_data.reshape(train_data.shape[0], -1), axis=1).reshape(train_data.shape)
    test_data = normalize(test_data.reshape(test_data.shape[0], -1), axis=1).reshape(test_data.shape)

    return train_data, train_targets, test_data, test_targets

def compute_accuracy(predictions, targets):
    accuracy = np.sum(np.array(predictions) == np.array(targets)) / len(predictions)
    return accuracy

def run_experiment(model_name, manipulation_info, num_train, savename=None):
    """ Trains and tests using train_model_function and evaluate_model_function
    arguments, and saves results. """
    min_faces_per_person = 20

    if manipulation_info.type == "dfi":
        train_data, train_targets, test_data, test_targets = get_lfw_dfi_dataset(min_faces_per_person, num_train, manipulation_info)
    else:
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

def download_images():
    """
    Download image datasets if necessary.
    Based on: https://github.com/paulu/deepfeatinterp/blob/master/demo1.py
    """
    if not os.path.exists('images/dfi'):
        url='https://www.dropbox.com/s/km3rnco93frlt0b/dfi.tar.gz?dl=1'
        subprocess.check_call(['wget',url,'-O','dfi.tar.gz'])
        subprocess.check_call(['tar','xzf','dfi.tar.gz'])
        subprocess.check_call(['rm','dfi.tar.gz'])
    if not os.path.exists('images/lfw_aegan'):
        url='https://www.dropbox.com/s/isz4ske2kheuwgr/lfw_aegan.tar.gz?dl=1'
        subprocess.check_call(['wget',url,'-O','lfw_aegan.tar.gz'])
        subprocess.check_call(['tar','xzf','lfw_aegan.tar.gz'])
        subprocess.check_call(['rm','lfw_aegan.tar.gz'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Only run experiments with this model")
    parser.add_argument("--manipulation", help="Only run experiments with this manipulation")
    parser.add_argument("--num-train", type=int, help="Only run experiments with this number of training images")
    parser.add_argument("--rerun", action="store_true", help="Re-run experiments (does not remove from .csv file)")
    args = parser.parse_args()

    download_images()

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
        ManipulationInfo("dfi", {"transform": "Senior"}),
        ManipulationInfo("dfi", {"transform": "Mustache"}),
    ]
    num_trains = [3, 10, 15, 19]

    if args.model:
        if args.model not in model_names:
            raise Exception("Unrecognized model name")
        model_names = [model for model in model_names if model == args.model]
    if args.manipulation:
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
            results = pd.DataFrame(columns=COLUMNS)
        else:
            results = pd.read_csv(save_path, float_precision='round_trip')

        # Run experiments
        with open(save_path, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)

            for manipulation_info in manipulation_infos:
                for model_name in model_names:
                    params_tuple = (manipulation_info.type, str(manipulation_info.parameters), model_name)

                    # Skip completed experiments
                    loc = (results["Manipulation Type"] == params_tuple[0]) & (results["Manipulation Parameters"] == params_tuple[1]) & (results["Recognition Algorithm"] == params_tuple[2])
                    assert(len(results[loc]) <= 1)
                    if len(results[loc]) > 0 and not args.rerun:
                        print("Skipping: %s" % str(params_tuple))
                        continue

                    # Clear any duplicates
                    results = results[~loc]

                    print("Running: %s" % str(params_tuple))
                    try:
                        results = results.append(run_experiment(model_name, manipulation_info, num_train), ignore_index=True)
                        results.to_csv(save_path, index=False)
                    except Exception as e:
                        print("Error:", e)
