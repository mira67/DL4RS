"""
Deep NN Models
Add customized models in this file
Author: Qi Liu
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils

"""
Models
"""
# basic dense serial NN
def model_1(x_train,y_train,x_test,y_test,p):
    model = Sequential()
    model.add(Dense(100, input_dim=p['fea_num']))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(p['out_num']))
    #sgd = SGD(lr=0.001, clipnorm=1.)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def model_2(x_train,y_train,x_test,y_test,p):
    model = Sequential()
    model.add(Dense(3, input_dim=p['fea_num']))
    model.add(Activation('tanh'))
    model.add(Dense(6))
    model.add(Activation('sigmoid'))
    model.add(Dense(p['out_num']))
    #sgd = SGD(lr=0.001, clipnorm=1.)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train_model(x_train,y_train,x_test,y_test,p):
    if p['model_id'] == 1:
        model = model_1(x_train,y_train,x_test,y_test,p)
    elif p['model_id'] == 2:
        model = model_2(x_train,y_train,x_test,y_test,p)
    #evaluate model
    history = model.fit(train_x, train_y,
              nb_epoch=p['iters'],
              batch_size=p['bsize'],
              verbose = 0,validation_split=0.10)
    #visualize results
    predicted_mpf = model.predict(test_x)*100
    test_ytr = test_y*100
    return predicted, gtest, model
