# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:00:44 2018

@author: gregoryvladimir.TRN
"""

import numpy as np
import os
import matplotlib.image as img
from keras.preprocessing import image
from shutil import copyfile, rmtree
import string
import stat

def convert_to_one_hot(class_number, num_classes=38):
    
    one_hot = np.zeros((num_classes))
    one_hot[class_number] = 1
    
    return one_hot

def vectorize_image(image_path, target_size=(299, 299)):
    
    pil_image = image.load_img(image_path, target_size=(299, 299))
    image_matrix = image.img_to_array(pil_image)
    
    return image_matrix
   
def shuffle_X_Y(X_matrix, Y_matrix):
    
    assert len(X_matrix) == len(Y_matrix)
    p = np.random.permutation(len(X_matrix))
    
    return X_matrix[p], Y_matrix[p]

def loop_through_and_rename(large_directory='data/crowdai_train/crowdai/', num_classes=38, train_set_ratio=0.8):
    
    train_dump_dir_path = os.path.join(large_directory, "trainingdump")
    validation_dump_dir_path = os.path.join(large_directory, "validationdump")
    
    if not os.path.exists(train_dump_dir_path):
        os.makedirs(train_dump_dir_path)
    
    if not os.path.exists(validation_dump_dir_path):
        os.makedirs(validation_dump_dir_path)
        
    if len(os.listdir(train_dump_dir_path)) != 0:
        raise ValueError("Files already exist in traindump and validationdump directories. Please delete these directories or all of the files in them and re-run.")
        
    for i in range(num_classes):
        current_dir = large_directory + "c_" + str(i)
        train_or_validation = np.random.uniform()
        for filename in os.listdir(current_dir):
            cur_image_path = os.path.join(current_dir, filename)
            

            random_name = ''.join(np.random.choice(list(string.ascii_uppercase + string.digits)) for _ in range(20)) + "_c_" + str(i) + ".JPG"
            
            if train_or_validation <= train_set_ratio:
                copyfile(cur_image_path, os.path.join(train_dump_dir_path, random_name))
                
            else:
                copyfile(cur_image_path, os.path.join(validation_dump_dir_path, random_name))
        
            
def create_features_list(directory):
    
    return os.listdir(directory)

def create_labels_list(features_list):
    
    ret = []
    
    for item in features_list:
        if item[-6].isdigit():
            ret.append(int(item[-6:-4]))
        else:
            ret.append(int(item[-5]))
            
    return ret
    