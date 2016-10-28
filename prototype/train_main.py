"""
Automated Algorithm/Data Testing Flow with Keras.
~~~
Best to put train/validate/test data into pickle data format, for easier
access for NN model
~~~
To-Do: format of y data
~~~
Author: Qi Liu
Date: Sep 20th, 2016
Version: 0.1
Update:Oct 4th
"""

from __future__ import print_function
import os
import ConfigParser
from get_data import read_data  as rd
from prj_util import ConfigSectionMap as cfgmap
import numpy as np
from nn_models import basicMPL
from keras.models import load_model
from keras import backend as K

"""
Default Configuration
"""
# data, result path
data_path = '/home/mirabot/googledrive/deeplearning/seaiceprj/'
result_path = '/home/mirabot/googledrive/deeplearning/seaiceprj/'
# training parameters
batch_size = 20 #128
nb_classes = 3 #10
nb_epoch = 3
# input image dimensions
img_rows, img_cols = 200, 200
color_dim = 3
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (4, 4)
# convolution kernel data_path
kernel_size = (3,3)

"""
Parse user configuration file
"""
config = ConfigParser.ConfigParser()
config.read("config.ini")
#config.sections()
data_path = cfgmap(config,"Workspace")['data_path']
result_path = cfgmap(config,"Workspace")['result_path']
# training parameters
img_rows = config.getint('Settings','img_rows')
img_cols = config.getint('Settings','img_cols')
color_dim = config.getint('Settings','color_dim')

batch_size = config.getint('Settings','batch_size')
nb_classes = config.getint('Settings','nb_classes')
nb_epoch = config.getint('Settings','nb_epoch')
# model
verbose_on = config.getboolean('Model','verbose_on')

# Import data
x_train,y_train,x_validate,y_validate,x_test,y_test,input_shape = rd(data_path,config,K)

# Test data
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_validate.shape[0],'validate samples')
print(x_test.shape[0], 'test samples')

"""
Train model and Save Model parameters
"""
model = basicMPL(input_shape)
# train the model
model.fit(x_train, y_train,
          batch_size=batch_size, nb_epoch=nb_epoch,verbose=verbose_on,
          validation_data=(x_valid, y_valid))
# Quick Evaluation
score = model.evaluate(x_test, y_test, verbose=1)
# Save model
model.save(result_path+'basicCNN.h5')  # creates a HDF5 file
