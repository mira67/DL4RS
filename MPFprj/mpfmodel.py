"""
Deep NN Models
Add customized models in this file
Author: Qi Liu
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.regularizers import l2, activity_l2
from keras.constraints import maxnorm
"""
Models
"""
# basic dense serial NN
def model_1(x_train,y_train,x_test,y_test,p):
    model = Sequential()
    model.add(Dense(100, input_dim=p['fea_num']))
    model.add(Activation('tanh'))
    model.add(Dropout(0.4))
    model.add(Dense(50))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(p['out_num']))
    #sgd = SGD(lr=0.001, clipnorm=1.)
    #adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def model_2(x_train,y_train,x_test,y_test,p):
    model = Sequential()
    model.add(Dense(56, input_dim=p['fea_num']))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(28))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(28))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('tanh'))
    model.add(Dense(p['out_num']))
    #sgd = SGD(lr=0.001, clipnorm=1.)#0.9,0.999
    #adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def model_3(x_train,y_train,x_test,y_test,p):
    model = Sequential()
    model.add(Dense(7, input_dim=p['fea_num']))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(p['out_num']))
    model.add(Activation('relu'))
    #sgd = SGD(lr=0.001, clipnorm=1.)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train_model(x_train,y_train,x_test,y_test,p):
    if p['model_id'] == 1:
        model = model_1(x_train,y_train,x_test,y_test,p)
    elif p['model_id'] == 2:
        model = model_2(x_train,y_train,x_test,y_test,p)
    elif p['model_id'] == 3:
        model = model_3(x_train,y_train,x_test,y_test,p)
    #evaluate model
    history = model.fit(x_train, y_train,
              nb_epoch=p['iters'],
              batch_size=p['bsize'],
              verbose = 0,validation_split=0.10)
    #visualize results
    predicted_mpf = model.predict(x_test)*100
    test_ytr = y_test*100
    return predicted_mpf, test_ytr, model, history
