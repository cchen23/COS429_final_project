# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 17:24:49 2017

@author: Cathy
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from sklearn.datasets import fetch_lfw_people
from collections import namedtuple

ManipulationInfo = namedtuple("ManipulationInfo", ["type", "parameters"])

def perform_manipulation(data, manipulation_info: ManipulationInfo):
    manipulation_type = manipulation_info.type
    manipulation_parameters = manipulation_info.parameters
    if manipulation_type == "none":
        return data
    elif manipulation_type == "occlude_lfw":
        occlusion_size = manipulation_parameters["occlusion_size"]
        return occlude_lfw_dataset(data, occlusion_size)
    elif manipulation_type == "radial_distortion":
        k = manipulation_parameters["k"]
        return radially_distort_lfw_dataset(data, k)
    elif manipulation_type == "blur":
        blurwindow_size = manipulation_parameters["blurwindow_size"]
        return blur_lfw_dataset(data, blurwindow_size)
    else:
        raise Exception("UNKNOWN MANIPULATION.")

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
    dataset_images = [np.reshape(radial_distortion(np.reshape(data[i], lfw_imageshape), distortion_array_i, distortion_array_j), lfw_datashape) for i in range(num_images)]
    return dataset_images

def radial_distortion(input_image, distortion_array_i, distortion_array_j):
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

def blur_lfw_dataset(data, blurwindow_size):
    """Returns a list of blurred images. Takes dataset.data from lfw as input."""
    num_images = data.shape[0]
    lfw_imageshape = (62,47)
    lfw_datashape = data[0].shape
    dataset_images = [np.reshape(blur(np.reshape(data[i], lfw_imageshape), blurwindow_size), lfw_datashape) for i in range(num_images)]
    return dataset_images

def blur(input_image, blurwindow_size):
    return ndimage.percentile_filter(input_image, -50, blurwindow_size)

def blur_slow(input_image, blurwindow_size):
    blurred_image = np.empty(input_image.shape)
    blurwindow_halflength = blurwindow_size / 2
    image_imax, image_jmax = input_image.shape
    for i in range(image_imax):
        for j in range(image_jmax):
            blurwindow_imin = int(max(0, i-blurwindow_halflength))
            blurwindow_jmin = int(max(0, j-blurwindow_halflength))
            blurwindow_imax = int(min(image_imax, i+blurwindow_halflength))
            blurwindow_jmax = int(min(image_jmax, j+blurwindow_halflength))
            blurred_image[i,j] = np.mean(input_image[blurwindow_imin:blurwindow_imax,blurwindow_jmin:blurwindow_jmax])
    return blurred_image
            
if __name__ == "__main__":
    min_faces_per_person = 20
    lfw_shape = (62,47)
    cmap="gray"
    dataset = fetch_lfw_people(min_faces_per_person=min_faces_per_person)
    test_image = np.reshape(dataset.data[0], lfw_shape)
    fig = plt.imshow(test_image, cmap=cmap)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig("../figures/manipulationdemo_none")
    plt.close()

    # Test add_occlusion.
    print("Testing occlusion")
    occlusion_sizes = [20, 30, 40]
    num_occlusionsizes = len(occlusion_sizes)
    test_images_occluded = []
    for occlusion_size in occlusion_sizes:
        test_images_occluded.append(add_occlusion(test_image, occlusion_size))
    f, axarr = plt.subplots(1, num_occlusionsizes, sharey=True)
    for i in range(num_occlusionsizes):
        axarr[i].imshow(test_images_occluded[i], cmap=cmap)
        axarr[i].set_title("Window Size %d" % occlusion_sizes[i])
        axarr[i].axis('off')
    plt.savefig("../figures/manipulationdemo_occlusion")
    plt.close()

    # Test radial distortion.
    print("Testing radial distortion")
    ks = [0.00015, 0.0003, 0.0005]
    num_ks = len(ks)
    test_images_pincushion = []
    test_images_barrel = []
    for k in ks:
        barrel_distortion_array_i, barrel_distortion_array_j = create_radial_distortion_array(k, lfw_shape)
        test_images_barrel.append(radial_distortion(test_image, barrel_distortion_array_i, barrel_distortion_array_j))
        pincushion_distortion_array_i, pincushion_distortion_array_j = create_radial_distortion_array(-k, lfw_shape)
        test_images_pincushion.append(radial_distortion(test_image, pincushion_distortion_array_i, pincushion_distortion_array_j))

    f, axarr = plt.subplots(2, num_ks, sharey=True)
    for i in range(num_ks):
        axarr[0,i].imshow(test_images_barrel[i], cmap=cmap)
        axarr[1,i].imshow(test_images_pincushion[i], cmap=cmap)
        axarr[0,i].tick_params(bottom='off',top='off',left='off',right='off',labelbottom='off',labeltop='off',labelleft='off',labelright='off')
        axarr[1,i].tick_params(bottom='off',top='off',left='off',right='off',labelbottom='off',labeltop='off',labelleft='off',labelright='off')
        axarr[0,i].set_frame_on(False)
        axarr[1,i].set_frame_on(False)
        axarr[0,i].set_title("k=%.5f" % ks[i])
    axarr[0,0].set_ylabel("Barrel")
    axarr[1,0].set_ylabel("Pincushion")
    plt.savefig("../figures/manipulationdemo_radial")
    plt.close()

    # Test blur.
    print("Testing blur")
    blurwindow_sizes = [5, 10]
    num_blurwindowsizes = len(blurwindow_sizes)
    test_images_blurred = []
    for blurwindow_size in blurwindow_sizes:
        dataset = fetch_lfw_people(min_faces_per_person=min_faces_per_person)
        test_images_blurred.append(blur(test_image, blurwindow_size))

    f, axarr = plt.subplots(1, num_blurwindowsizes, sharey=True)
    for i in range(num_blurwindowsizes):
        axarr[i].imshow(test_images_blurred[i], cmap=cmap)
        axarr[i].set_title("Window size %d" % blurwindow_sizes[i])
        axarr[i].axis('off')
    plt.savefig("../figures/manipulationdemo_blur")
    plt.close()
    
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
