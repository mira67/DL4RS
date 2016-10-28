
"""
Assumption: image data are preprocessed (i.e,mask, missing data processing)
and well organized in to train/validate/test partitions. If not, need
additional modules to prepare train/validate/test data
For example, get train data have pixel to pixel, or batch to batch (x,y)
paired data, either integrate in single one-one mutilayer image or seperate two files
~~~
Prepare Training, Validation and Testing Data [train, validate,test]
No shuffle yet, follow 'train,validate,test' predecided partitions
~Assume seperate (x,y) files
~~~
Author: Qi Liu
"""
from __future__ import absolute_import
from prj_util import ConfigSectionMap as cfgmap
import os
from PIL import Image
import glob
import re
import ConfigParser
import numpy as np

"""
Function to sort files in order
"""
def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def read_data(data_path, config,K):
    # image configuration
    img_format = cfgmap(config,"Settings")['img_format']
    img_scale = config.getint('Settings','img_scale')
    img_scalar = config.getint('Settings','img_scalar')

    img_rows = config.getint('Settings','img_rows')
    img_cols = config.getint('Settings','img_cols')
    color_dim = config.getint('Settings','color_dim')

    # get daself, ta
    train_path = data_path+'train/'
    validate_path = data_path+'validate/'
    test_path = data_path+'test/'

    x_train_list = sorted(glob.glob(train_path+'*_x.'+img_format),key=numericalSort)
    x_train_array = np.array( [np.array(Image.open(x_train_list[i]), 'f') for i in range(len(x_train_list))])

    #y_train_list = sorted(glob.glob(train_path+'*_y.'+img_format),key=numericalSort)
    #y_train_array = np.array( [np.array(Image.open(y_train_list[i]), 'f') for i in range(len(imagePath))])

    x_validate_list = sorted(glob.glob(train_path+'*_x.'+img_format),key=numericalSort)
    x_validate_array = np.array( [np.array(Image.open(x_validate_list[i]), 'f') for i in range(len(x_validate_list))])

    #y_validate_list = sorted(glob.glob(train_path+'*_y.'+img_format),key=numericalSort)
    #y_validate_array = np.array( [np.array(Image.open(y_validate_list[i]), 'f') for i in range(len(imagePath))])

    x_test_list = sorted(glob.glob(train_path+'*_x.'+img_format),key=numericalSort)
    x_test_array = np.array( [np.array(Image.open(x_test_list[i]), 'f') for i in range(len(x_test_list))])

    #y_test_list = sorted(glob.glob(train_path+'*_y.'+img_format),key=numericalSort)
    #y_test_array = np.array( [np.array(Image.open(y_test_list[i]), 'f') for i in range(len(imagePath))])

    # Reformat and Obtain input shape for NN model
    if K.image_dim_ordering() == 'th':
        x_train = x_train_array.reshape(x_train_array.shape[0], color_dim, img_rows, img_cols)
        x_validate = x_validate_array.reshape(x_validate_array.shape[0], color_dim, img_rows, img_cols)
        x_test = x_test_array.reshape(x_test_array.shape[0], color_dim, img_rows, img_cols)
        input_shape = (color_dim, img_rows, img_cols)
    else:
        x_train = x_train_array.reshape(x_train_array.shape[0], img_rows, img_cols, color_dim)
        x_train = x_validate_array.reshape(x_validate_array.shape[0], img_rows, img_cols, color_dim)
        x_test = x_test_array.reshape(x_test_array.shape[0], img_rows, img_cols, color_dim)
        input_shape = (img_rows, img_cols, color_dim)

    x_train = x_train.astype('float32')
    x_validate = x_validate.astype('float32')
    x_test = x_test.astype('float32')

    if img_scale:
        x_train /= img_scalar
        x_test /= img_scalar
        x_validate /= img_scalar

    #fake data
    y_train = 0
    y_validate = 1
    y_test = 2

    return x_train,y_train,x_validate,y_validate,x_test,y_test,input_shape
