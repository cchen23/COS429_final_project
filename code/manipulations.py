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
    elif manipulation_type == "radial_distortion":
        k = manipulation_parameters[0]
        return radially_distort_lfw_dataset(data, k)
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

#def create_radial_distortion_array(k1, input_image_shape):
#    # Using notation/equations from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4934233/pdf/sensors-16-00807.pdf
#    i_max, j_max = input_image_shape
#    i0 = int(i_max / 2)
#    j0 = int(j_max / 2)
#    distortion_array_i = np.zeros(input_image_shape, dtype='int')
#    distortion_array_j = np.zeros(input_image_shape, dtype='int')
#    for i in range(i_max):
#        for j in range(j_max):
#            i_bar = i - i0
#            j_bar = j - j0
#            r_squared = (i_bar * i_bar) + (j_bar * j_bar)
#            i_new = round(i + i_bar * k1 * r_squared)
#            j_new = round(j + j_bar * k1 * r_squared)
#            if i_new < i_max and j_new < j_max and i_new >= 0 and j_new >= 0:
#                distortion_array_i[i_new, j_new] = i
#                distortion_array_j[i_new, j_new] = j
#    return distortion_array_i, distortion_array_j

def radially_distort_lfw_dataset(data, k):
    lfw_imageshape = (62, 47)
    distortion_array_i, distortion_array_j = create_radial_distortion_array(k, lfw_imageshape)
    lfw_datashape = data[0].shape
    num_images = data.shape[0]
    dataset_images = [np.reshape(radial_distortion(distortion_array_i, distortion_array_j, np.reshape(data[i], lfw_imageshape)), lfw_datashape) for i in range(num_images)]
    return dataset_images

def radial_distortion(distortion_array_i, distortion_array_j, input_image):
    distorted_image = np.empty(distortion_array_i.shape)
    for i in range(distorted_image.shape[0]):
        for j in range(distorted_image.shape[1]):
            input_i = distortion_array_i[i, j]
            input_j = distortion_array_j[i, j]
            distorted_image[i,j] = input_image[input_i, input_j]
    return distorted_image

def create_radial_distortion_array(k, input_image_shape):
    # http://sprg.massey.ac.nz/pdfs/2003_IVCNZ_408.pdf
    # x_d = x_u / (1+kr_d^2)
    # Negative k for pincushion, positive k for barrel.
    i_max, j_max = input_image_shape
    i0 = int(i_max / 2)
    j0 = int(j_max / 2)
    distortion_array_i = np.zeros(input_image_shape, dtype='int')
    distortion_array_j = np.zeros(input_image_shape, dtype='int')
    for i in range(i_max):
        for j in range(j_max):
            i_bar = i - i0
            j_bar = j - j0
            r_squared = (i_bar * i_bar) + (j_bar * j_bar)
            i_input = i / (1+k*r_squared)
            j_input = j / (1+k*r_squared)
            if i_input < i_max and j_input < j_max and i_input >= 0 and j_input >= 0:
                distortion_array_i[i, j] = i_input
                distortion_array_j[i, j] = j_input
    return distortion_array_i, distortion_array_j


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
    occluded_images = perform_manipulation(dataset.data, ("occlude_lfw", occlusion_size))
    num_examples = 5
    f, axarr = plt.subplots(num_examples, 2)
    for i in range(num_examples):
        axarr[i,0].imshow(np.reshape(dataset.data[i], lfw_shape))
        axarr[i,1].imshow(np.reshape(occluded_images[i], lfw_shape))
    axarr[0,0].set_title("Original images.")
    axarr[0,1].set_title("Occluded images.")

    # Test radial distortion.
    min_faces_per_person = 20
    occlusion_size = 20
    lfw_shape = (62,47)
    k1 = 0.0015
    dataset = fetch_lfw_people(min_faces_per_person=min_faces_per_person)
    test_image = np.reshape(dataset.data[0], lfw_shape)
    barrel_distortion_array_i, barrel_distortion_array_j = create_radial_distortion_array(k1, lfw_shape)
    test_image_barrel = radial_distortion(barrel_distortion_array_i, barrel_distortion_array_j, test_image)
    pincushion_distortion_array_i, pincushion_distortion_array_j = create_radial_distortion_array(-k1, lfw_shape)
    test_image_pincushion = radial_distortion(pincushion_distortion_array_i, pincushion_distortion_array_j, test_image)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    ax1.imshow(test_image)
    ax2.imshow(test_image_barrel)
    ax3.imshow(test_image_pincushion)
    ax1.set_title("Original image.")
    ax2.set_title("Barrel distortion image.")
    ax3.set_title("Pincushion distortion image.")
