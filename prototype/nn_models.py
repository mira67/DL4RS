"""
Deep NN Models
Add customized models in this file
Author: Qi Liu
Upate:
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

"""
Models
"""
# basic dense serial NN
def basicMPL(input_sp):
    # linear or tanh activation works slightly
    model = Sequential()
    model.add(Dense(10, input_shape=input_sp))
    model.add(Activation('linear'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

# basic CNN
def basicCNN(input_sp):
    return model

# temporal DL
def tempDL(input_sp):
    return model
