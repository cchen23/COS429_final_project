# Get LFW dataset images, save as .mat file to use in MATLAB.

import scipy
from sklearn.datasets import fetch_lfw_people
from scipy import ndimage

def get_lfw_dataset(min_faces_per_person):
    dataset = fetch_lfw_people(
        min_faces_per_person=min_faces_per_person, 
        color=True, 
        slice_=(slice(0, 250, None), slice(0, 250, None)), 
        resize=1)
    data = dataset.images
    
    return data

def main():
    min_faces_per_person = 20
    data = get_lfw_dataset(min_faces_per_person)
    scipy.io.savemat('faces.mat', mdict={'data': data})
    print(data.shape)
    print(data[0].shape)

main()