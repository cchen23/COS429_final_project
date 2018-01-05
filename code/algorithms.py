# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 15:39:53 2017

@author: Cathy
"""
#import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.decomposition import SparseCoder

# Helper functions.
def reduce_dimensions(train_data):
    n_components=None # TODO: Test with different numbers of components?
    pca = PCA(n_components=n_components)
    pca.fit(train_data)
    train_data = pca.transform(train_data)
    return train_data, pca

# Training functions.
def train_pca(train_data, train_targets):
    # Train data is num_people x num_dims.
    # TODO: What if more than one training example per person? Average weights? Use each?
    
    # Implemented based on: http://www.vision.jhu.edu/teaching/vision08/Handouts/case_study_pca1.pdf
    C = np.dot(train_data, np.transpose(train_data))
    u, s, v = np.linalg.svd(C)
    eigenfaces = np.dot(np.transpose(train_data), u) # num_dims x num_eigenfaces. Each column is an eigenface.
    train_weights = np.dot(np.transpose(eigenfaces), np.transpose(train_data)) # num_people x num_eigenfaces. Each column is a set of weights.
    
#    for i in range(5):
#        eigenface = eigenfaces[:,i]
#        plt.figure(i)
#        plt.imshow(np.reshape(eigenface, (62,47)))
    return eigenfaces, train_weights, train_targets

def train_sparserepresentation(train_data, train_targets):
    # Implemented based on: https://people.eecs.berkeley.edu/~yang/paper/face_chapter.pdf (1.2)
    coder = SparseCoder(train_data, transform_algorithm='omp')
    return coder, train_targets

def train_sparserepresentation_dimension_reduction(train_data, train_targets):
    # Implemented based on: https://people.eecs.berkeley.edu/~yang/paper/face_chapter.pdf (1.3)
    train_data, pca = reduce_dimensions(train_data)
    coder = SparseCoder(train_data, transform_algorithm='omp')
    return coder, pca, train_targets

def train_sparserepresentation_combinedl1(train_data, train_targets):
    # Implemented based on: https://people.eecs.berkeley.edu/~yang/paper/face_chapter.pdf (1.4)
    # train_data is nxm
    train_data, pca = reduce_dimensions(train_data)
    I = np.eye(train_data.shape[1])
    dictionary = np.concatenate((train_data, I))
    coder = SparseCoder(dictionary, transform_algorithm='omp')
    return coder, pca, train_targets

# Prediction functions.
def predict_pca(model, test_data):
    eigenfaces, train_weights, train_targets = model
    num_test_faces = test_data.shape[0]
    test_predictions = []
    weight_length = train_weights.shape[0]
    
    test_weights = np.dot(np.transpose(eigenfaces), np.transpose(test_data))
    for i in range(num_test_faces):
        test_weight = np.reshape(test_weights[:,i], (weight_length, 1))
        diffs = train_weights - test_weight
        diff_norms = np.linalg.norm(diffs, axis=0)
        test_prediction = train_targets[np.argmin(diff_norms)]
        test_predictions.append(test_prediction)
    
    return test_predictions 

def predict_sparserepresentation(model, test_data):
    coder, train_targets = model
    weights = coder.transform(test_data)
    train_labels = np.unique(train_targets)
    num_test_examples = test_data.shape[0]
    test_predictions = []

    for i in range(num_test_examples):
        example_weights = weights[i,:]
        max_label_weight = -float("inf")
        max_label = -1
        for label in train_labels:
            label_indices = np.where(train_targets == label)
            label_weights = np.sum(example_weights[label_indices])
            if label_weights > max_label_weight:
                max_label_weight = label_weights
                max_label = label
        test_predictions.append(max_label)

    return test_predictions

def predict_sparserepresentation_dimension_reduction(model, test_data):
    coder, pca, train_targets = model
    test_data = pca.transform(test_data)
    weights = coder.transform(test_data)
    train_labels = np.unique(train_targets)
    num_test_examples = test_data.shape[0]
    test_predictions = []

    for i in range(num_test_examples):
        example_weights = weights[i,:]
        max_label_weight = -float("inf")
        max_label = -1
        for label in train_labels:
            label_indices = np.where(train_targets == label)
            label_weights = np.sum(example_weights[label_indices])
            if label_weights > max_label_weight:
                max_label_weight = label_weights
                max_label = label
        test_predictions.append(max_label)

    return test_predictions

def predict_sparserepresentation_combinedl1(model, test_data):
    coder, pca, train_targets = model
    test_data = pca.transform(test_data)
    weights = coder.transform(test_data)
    train_labels = np.unique(train_targets)
    num_train_examples = len(train_targets)
    num_test_examples = test_data.shape[0]
    test_predictions = []

    for i in range(num_test_examples):
        example_weights = weights[i,:num_train_examples] # Don't take into account weights from e.
        max_label_weight = -float("inf")
        max_label = -1
        for label in train_labels:
            label_indices = np.where(train_targets == label)
            label_weights = np.sum(example_weights[label_indices])
            if label_weights > max_label_weight:
                max_label_weight = label_weights
                max_label = label
        test_predictions.append(max_label)

    return test_predictions

ALGORITHM_MAP = {
    "PCA": (train_pca, predict_pca),
    "Sparse Representation": (train_sparserepresentation, predict_sparserepresentation),
    "Sparse Representation Dimension Reduction": (train_sparserepresentation_dimension_reduction, predict_sparserepresentation_dimension_reduction),
    "Sparse Representation Combined l1": (train_sparserepresentation_combinedl1, predict_sparserepresentation_combinedl1),
}

def train(model_name, train_data, train_targets):
    return ALGORITHM_MAP[model_name][0](train_data=train_data, train_targets=train_targets)

def predict(model_name, model, test_data):
    return ALGORITHM_MAP[model_name][1](model=model, test_data=test_data)
