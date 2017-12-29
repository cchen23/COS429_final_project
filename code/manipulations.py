# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 17:24:49 2017

@author: Cathy
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_lfw_people

def perform_manipulation(data, manipulation_info):
    manipulation_type = manipulation_info[0]
    manipulation_parameters = manipulation_info[1:]
    if manipulation_type == "none":
        return data
    elif manipulation_type == "occlude_lfw":
        occlusion_size = manipulation_parameters[0]
        return occlude_lfw_dataset(data, occlusion_size)
    else:
        print("UNKNOWN MANIPULATION.")

# Manipulation definitions.
def occlude_lfw_dataset(data, occlusion_size):
    """Returns a list of occluded images. Takes dataset.data from lfw as input."""
    num_images = data.shape[0]
    lfw_imageshape = (62,47)
    lfw_datashape = data[0].shape
    dataset_images = [np.reshape(add_occlusion(np.reshape(data[i], lfw_imageshape), occlusion_size), lfw_datashape) for i in range(num_images)]
    return dataset_images

def add_occlusion(input_image, occlusion_size):
    """Randomly selects an occlusion_size-by-occlusion_size square in the image
    and sets the pixels to random values between 0 and 256."""
    max_value = 256
    max_i, max_j = input_image.shape
    input_image = np.copy(input_image)
    start_i = np.random.randint(0, max_i-occlusion_size)
    start_j = np.random.randint(0, max_j-occlusion_size)
    occlusion_square = np.random.rand(occlusion_size, occlusion_size)*max_value
    input_image[start_i:start_i+occlusion_size,start_j:start_j+occlusion_size]=occlusion_square
    return input_image

if __name__ == "__main__":
    # Test add_occlusion.
    min_faces_per_person = 20
    occlusion_size = 20
    lfw_shape = (62,47)
    dataset = fetch_lfw_people(min_faces_per_person=min_faces_per_person)
    test_image = np.reshape(dataset.data[0], lfw_shape)
    test_image_occluded = add_occlusion(test_image, occlusion_size)
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(test_image)
    ax2.imshow(test_image_occluded)
    ax1.set_title("Original image.")
    ax2.set_title("Occluded image.")
    
    # Test occlude_lfw_dataset.
    occluded_images = occlude_lfw_dataset(dataset.data, occlusion_size)
    num_examples = 5
    f, axarr = plt.subplots(num_examples, 2)
    for i in range(num_examples):
        axarr[i,0].imshow(np.reshape(dataset.data[i], lfw_shape))
        axarr[i,1].imshow(np.reshape(occluded_images[i], lfw_shape))
    axarr[0,0].set_title("Original images.")
    axarr[0,1].set_title("Occluded images.")
    
    # Test perform_manipulations.
    occluded_images = perform_manipulations(dataset.data, ("occlude_lfw", occlusion_size))
    num_examples = 5
    f, axarr = plt.subplots(num_examples, 2)
    for i in range(num_examples):
        axarr[i,0].imshow(np.reshape(dataset.data[i], lfw_shape))
        axarr[i,1].imshow(np.reshape(occluded_images[i], lfw_shape))
    axarr[0,0].set_title("Original images.")
    axarr[0,1].set_title("Occluded images.")