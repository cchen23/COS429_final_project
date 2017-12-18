# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 15:39:53 2017

@author: Cathy
"""
import matplotlib.pyplot as plt
import numpy as np

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
    