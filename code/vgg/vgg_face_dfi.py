
# coding: utf-8

# # COS 429 Final Project
# ## VGG Face - DFI Manipulations
# 
# AWS setup:
# - `source activate theano_p36`
# - `conda install -c anaconda pillow`
# - `conda install h5py`
# - `conda install scikit-learn`
# - `jupyter notebook`
# - `scp -i cos429.pem *.py ubuntu@...:~/cos429/`
# 
# Download the VGG-FACE pre-trained weights for Keras here: https://drive.google.com/file/d/0B4ChsjFJvew3NkF0dTc1OGxsOFU/view.
# 
# Before stopping the instance, remember to download the latest .ipynb file for the GitHub. Terminate the instance to delete all files.

# In[1]:


# Only run once at start of program
import os
os.chdir('..')


# In[2]:


import numpy as np
import matplotlib.pyplot as plt


import time
import os
import sys  
# os.environ['THEANO_FLAGS'] = "device=gpu1"    
# import theano
import pandas as pd

from keras.models import Model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation

from keras import backend as K
K.set_image_dim_ordering('th')

from PIL import Image


# In[3]:


weights_path = 'vgg/vgg-face-keras.h5'


# In[4]:


# This network architecture is derived from Table 3 of the CNN described in Parkhi et al. 
# and based on Keras code provided in https://gist.github.com/EncodeTS/6bbe8cb8bebad7a672f0d872561782d9

def vgg_face(weights_path=None):
    img = Input(shape=(3, 224, 224))

    pad1_1 = ZeroPadding2D(padding=(1, 1))(img)
    conv1_1 = Convolution2D(64, (3, 3), activation='relu', name='conv1_1')(pad1_1)
    pad1_2 = ZeroPadding2D(padding=(1, 1))(conv1_1)
    conv1_2 = Convolution2D(64, (3, 3), activation='relu', name='conv1_2')(pad1_2)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1_2)

    pad2_1 = ZeroPadding2D((1, 1))(pool1)
    conv2_1 = Convolution2D(128, (3, 3), activation='relu', name='conv2_1')(pad2_1)
    pad2_2 = ZeroPadding2D((1, 1))(conv2_1)
    conv2_2 = Convolution2D(128, (3, 3), activation='relu', name='conv2_2')(pad2_2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2_2)

    pad3_1 = ZeroPadding2D((1, 1))(pool2)
    conv3_1 = Convolution2D(256, (3, 3), activation='relu', name='conv3_1')(pad3_1)
    pad3_2 = ZeroPadding2D((1, 1))(conv3_1)
    conv3_2 = Convolution2D(256, (3, 3), activation='relu', name='conv3_2')(pad3_2)
    pad3_3 = ZeroPadding2D((1, 1))(conv3_2)
    conv3_3 = Convolution2D(256, (3, 3), activation='relu', name='conv3_3')(pad3_3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3_3)

    pad4_1 = ZeroPadding2D((1, 1))(pool3)
    conv4_1 = Convolution2D(512, (3, 3), activation='relu', name='conv4_1')(pad4_1)
    pad4_2 = ZeroPadding2D((1, 1))(conv4_1)
    conv4_2 = Convolution2D(512, (3, 3), activation='relu', name='conv4_2')(pad4_2)
    pad4_3 = ZeroPadding2D((1, 1))(conv4_2)
    conv4_3 = Convolution2D(512, (3, 3), activation='relu', name='conv4_3')(pad4_3)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4_3)

    pad5_1 = ZeroPadding2D((1, 1))(pool4)
    conv5_1 = Convolution2D(512, (3, 3), activation='relu', name='conv5_1')(pad5_1)
    pad5_2 = ZeroPadding2D((1, 1))(conv5_1)
    conv5_2 = Convolution2D(512, (3, 3), activation='relu', name='conv5_2')(pad5_2)
    pad5_3 = ZeroPadding2D((1, 1))(conv5_2)
    conv5_3 = Convolution2D(512, (3, 3), activation='relu', name='conv5_3')(pad5_3)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2))(conv5_3)

    # These layers are used in the original VGG Face paper for their dataset of 2,622 individuals
    # The output of the previous layer is the 4096-dimensional face descriptor
    fc6 = Convolution2D(4096, (7, 7), activation='relu', name='fc6')(pool5)
    fc6_drop = Dropout(0.5)(fc6)
    fc7 = Convolution2D(4096, (1, 1), activation='relu', name='fc7')(fc6_drop)
    fc7_drop = Dropout(0.5)(fc7)
    fc8 = Convolution2D(2622, (1, 1), name='fc8')(fc7_drop)
    flat = Flatten()(fc8)
    out = Activation('softmax')(flat)

    model = Model(inputs=img, outputs=out)

    if weights_path:
        model.load_weights(weights_path)

    return model

# Returns model that for the 4096-dimensional face descriptor 
def partial_vgg_face():
    model = vgg_face(weights_path)
    layer_name = 'fc7'
    partial_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
    return partial_model


# In[5]:



from sklearn.datasets import fetch_lfw_people
from scipy.spatial.distance import cosine
from scipy import ndimage
from scipy.stats import mode
import h5py

import manipulations
import experiment
from manipulations import ManipulationInfo


# In[6]:


class TrainedModel:
    def __init__(self, num_train):
        self.num_train = num_train
        
        print('Loading model')
        self.model = partial_vgg_face()
        
        print('Loading training dataset')
        self.min_faces_per_person = 20
        self.train_data, self.train_targets, _, _ = experiment.get_lfw_dataset(self.min_faces_per_person, num_train, color=True, size=224)
        
        # Train
        print('Training')
        time1 = time.clock()
        self.num_faces = len(np.unique(self.train_targets))
        self.num_examples_per_face = int(len(self.train_targets) / self.num_faces)
        self.train_descriptors = self.get_descriptors(self.train_data)
        time2 = time.clock()
        self.train_time = time2 - time1
        print('Training time: %f' % self.train_time)

    # Prediction with nearest-k neighbors and cosine similarity
    def predict(self, test_descriptors, k):
        predictions = []

        for d in test_descriptors:
            # Use cosine similarity
            distances = [cosine(self.train_descriptors[i], d) for i in range(len(self.train_descriptors))]

            # The closest k vote instead of average
            closest_k = np.asarray(distances).argsort()[:k] // self.num_examples_per_face
            predictions.append(mode(closest_k).mode)
        return np.asarray(predictions).flatten()

    def predict_train(self, k):
        return self.predict(self.train_descriptors, k)

    def get_descriptors(self, data):
        data = data.transpose((0, 3, 1, 2))
        descriptors = self.model.predict(data, verbose=1)
        return np.squeeze(descriptors)

    # Alternative prediction method with average and L2 Euclidean distance
    # def predict_with_mean(train_descriptors, test_descriptors, num_examples_per_face, threshold=None):
    #     predictions = []
    #     mean_train_descriptors = np.mean(np.reshape(train_descriptors, (-1, num_examples_per_face, 4096)), axis=1)

    #     for d in test_descriptors:
    #         distances = [np.linalg.norm(mean_train_descriptors[i] - d) for i in range(len(mean_train_descriptors))]
    #         predictions.append(np.argmin(distances))
    #     return np.asarray(predictions)


# In[ ]:


def run_experiment(trained_model, manipulation_info):
    _, _, test_data, test_targets = experiment.get_lfw_dfi_dataset(trained_model.min_faces_per_person, trained_model.num_train, manipulation_info, color=True, size=224)
    test_descriptors = trained_model.get_descriptors(test_data)

    # Test
    print('Testing')
    time1 = time.clock()
    train_predictions_1 = trained_model.predict_train(1)
    train_predictions_3 = trained_model.predict_train(3)
    train_predictions_5 = trained_model.predict_train(5)
    train_predictions_7 = trained_model.predict_train(7)
    # train_accuracy = experiment.compute_accuracy(train_predictions, test_targets)
    train_accuracy_1 = experiment.compute_accuracy(train_predictions_1, trained_model.train_targets)
    train_accuracy_3 = experiment.compute_accuracy(train_predictions_3, trained_model.train_targets)
    train_accuracy_5 = experiment.compute_accuracy(train_predictions_5, trained_model.train_targets)
    train_accuracy_7 = experiment.compute_accuracy(train_predictions_7, trained_model.train_targets)
    
    # Predict test_descriptors
    test_predictions_1 = trained_model.predict(test_descriptors, 1)
    test_predictions_3 = trained_model.predict(test_descriptors, 3)
    test_predictions_5 = trained_model.predict(test_descriptors, 5)
    test_predictions_7 = trained_model.predict(test_descriptors, 7)
    # test_accuracy = experiment.compute_accuracy(test_predictions, test_targets)
    test_accuracy_1 = experiment.compute_accuracy(test_predictions_1, test_targets)
    test_accuracy_3 = experiment.compute_accuracy(test_predictions_3, test_targets)
    test_accuracy_5 = experiment.compute_accuracy(test_predictions_5, test_targets)
    test_accuracy_7 = experiment.compute_accuracy(test_predictions_7, test_targets)
    time2 = time.clock()
    test_time = time2 - time1
    
    train_accuracy = { "k=1": train_accuracy_1, 
                      "k=3": train_accuracy_3, 
                      "k=5": train_accuracy_5, 
                      "k=7": train_accuracy_7 }
    
    test_accuracy = { "k=1": test_accuracy_1, 
                      "k=3": test_accuracy_3, 
                      "k=5": test_accuracy_5, 
                      "k=7": test_accuracy_7 }
    
    # Print results.
    num_faces = len(np.unique(trained_model.train_targets))
    model_name = 'VGG_FACE'
    print("Manipulation info: %s" % str(manipulation_info))
    print("Recognition Algorithm: %s" % model_name)
    print("Number of distinct faces: %d" % num_faces)
    print("Chance rate: %f" % (1 / num_faces))
    print("Train accuracy: %s" % train_accuracy)
    print("Test accuracy: %s" % test_accuracy)
    print("Training Time: %s sec" % trained_model.train_time)
    print("Testing Time: %s sec" % test_time)
    print("\n")

    
    return {
        "Manipulation Type": manipulation_info.type,
        "Manipulation Parameters": manipulation_info.parameters,
        "Recognition Algorithm": model_name,
        "Min Faces Per Person": trained_model.min_faces_per_person,
        "Number of Distinct Faces": num_faces,
        "Chance Rate": (1 / num_faces),
        "Train Accuracy": train_accuracy,
        "Test Accuracy": test_accuracy,
        "Training Time": trained_model.train_time,
        "Testing Time": test_time,
    }


# In[ ]:


manipulation_infos = [
        ManipulationInfo("dfi", {"transform": "Senior"}),
        ManipulationInfo("dfi", {"transform": "Mustache"}),
    ]

num_trains = [3, 10, 15, 19]
for num_train in num_trains:
    print("Num training examples: %d" % num_train)
    save_path = "vgg_face_dfi_%d.csv" % num_train
    results = pd.DataFrame(columns=experiment.COLUMNS)
    trained_model = TrainedModel(num_train)
    for manipulation in manipulation_infos:
        stats = run_experiment(trained_model, manipulation)
        results.append(stats, ignore_index=True)
        print(stats)
        print()
    results.to_csv(save_path, index=False)
    print()
    print()


# In[ ]:


# # Graphs

# # Accuracy as a function of k for 3 training images
# # With and without preprocessing
# # Mean/Euclidean distance included as a line

# accuracy = np.array([0.8808600634473035, 0.83785689108212902, 0.83080719069439546, 0.79978850898836795])
# preprocessed_accuracy = np.array([0.96686640817765246, 0.96263658794501239, 0.94360239689813186, 0.9002467395135707])
# k = np.array([1, 3, 5, 7])

# fig, ax = plt.subplots()
# width = 0.35
# acc1 = ax.bar(k, accuracy, width, label='No preprocessing')
# acc2 = ax.bar(k+width, preprocessed_accuracy, width, label='Preprocessed')
# ax.plot([0, 8], [0.85442368699330273, 0.85442368699330273], 'k-', lw=1, linestyle='dashed')
# ax.set_xticks(k)
# ax.set_xlabel('$k$')
# ax.set_ylabel('Accuracy')
# ax.set_title('VGG_FACE accuracy on LFW with varying $k$ and 3 training images')
# legend = ax.legend(handles=[acc1, acc2], loc=4)
# plt.show()

