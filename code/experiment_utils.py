# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 15:39:53 2017

@author: Cathy
"""
import numpy as np

# Training functions.

def train_pca(train_data, train_target):
    # Implemented based on: http://www.vision.jhu.edu/teaching/vision08/Handouts/case_study_pca1.pdf
    A = np.transpose(train_data) # Using A to match notation in vision.jhu.edu explanation.
    C = np.cov(np.transpose(A))
    u, s, v = np.linalg.svd(C)
    eigenfaces = np.dot(A, u) # Each column is an eigenface.
    weights = np.dot(np.transpose(eigenfaces), A) # Each column is a set of weights.
    
    return weights, train_target

# Evaluation functions.
def evaluate_pca(model, test_data, test_target):
    pass