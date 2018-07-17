# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 10:28:24 2018

@author: gregoryvladimir.TRN
"""

import numpy as np
import keras
from preprocess import *
import os

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, features, labels, batch_size=32, dim=(299, 299), n_channels=3,
                 n_classes=38, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.features = features
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.features) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        features_temp = [self.features[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(features_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.features))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, features_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        path_to_image = "data/crowdai_train/crowdai/trainingdump/"

        # Generate data
        for i, ID in enumerate(features_temp):
            # Store sample
            X[i,] = vectorize_image(os.path.join(path_to_image, ID))

            # Store class
            y[i] = self.labels[i]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)