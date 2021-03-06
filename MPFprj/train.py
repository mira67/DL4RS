"""
Train Main, configure config.ini for data/model parameters
Author: Qi Liu
Date: 10/2016, updated 11/2016
"""
import os
import math
import timeit
import logging
from prjutil import read_config
from dbaccess import pdread
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.metrics import r2_score
from mpfmodel import train_model
from visdata import vis_train
import matplotlib.mlab as mlab

def train_kfold(X,Y,p):
    logging.info('kfold training...')
    kfold = KFold(n_splits=p['n_splits'], shuffle=True, random_state=1)#, random_state=1
    cvscores = []
    cvr = []
    r2 = []
    for train, test in kfold.split(X):
        predicted, gtest, model, history = train_model(X[train],Y[train],X[test],Y[test],p)
        score = model.evaluate(X[test], Y[test], batch_size=p['bsize'], verbose=0)
        mpf_mse = np.absolute(np.mean(predicted[:,0],axis=0)-np.mean(gtest[:,0],axis=0))/np.mean(gtest[:,0],axis=0)*100
        r = np.corrcoef(predicted[:,0], gtest[:,0])[1,0]
        cvr.append(r)
        cvscores.append(mpf_mse)
        r2.append(r2_score(gtest, predicted))
    return cvr, cvscores, r2, predicted, gtest, model, history

def train_all(X,Y,p):
    #np.random.seed(1)#for repeating results
    msk = np.random.rand(len(Y)) < 0.8
    X_train = X[msk]
    X_test = X[~msk]
    Y_train = Y[msk]
    Y_test = Y[~msk]
    cvscores = []
    cvr = []
    r2 = []
    predicted, gtest, model, history = train_model(X_train,Y_train,X_test,Y_test,p)
    score = model.evaluate(X_test, Y_test, batch_size=p['bsize'], verbose=0)
    mpf_mse = np.absolute(np.mean(predicted[:,0],axis=0)-np.mean(gtest[:,0],axis=0))/np.mean(gtest[:,0],axis=0)*100
    r = np.corrcoef(predicted[:,0], gtest[:,0])[1,0]
    cvr.append(r)
    cvscores.append(mpf_mse)
    r2.append(r2_score(gtest, predicted))
    return cvr, cvscores, r2,predicted, gtest, model, history

def main():
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-4s %(levelname)-4s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='training.log',
                    filemode='a')
    logging.info('Started Training')
    #read config
    p = read_config();
    logging.info('Training with Model: ' + str(p['model_id']))
    #get train/validation/test data
    df = pdread(p['train_sql'])
    df = df.replace(-9999, 0)
    # basen = 2
    # alpha = 0.8
    # beta = 0
    # julian_m = 197.5
    #
    # #gaussian kernel method
    # mu = 0
    # variance = 0.5
    # sigma = math.sqrt(variance)
    # # mayfilter = df['day']
    # # mayfilter[mayfilter<=153] = 0.5
    # # mayfilter[mayfilter>153] = 0
    # df['day'][df['day']>213] = alpha*mlab.normpdf((df['day'][df['day']>213]-julian_m)/31, mu, sigma) + beta
    # df['day'][df['day']<=213] = 1
    #
    # df['b1'] = df['day']*df['b1']

    print df.head()
    data = df.as_matrix()
    print('data size: ', data.shape)
    X = data[:,0:p['fea_num']]
    #X = normalize(X,norm='l2',axis=0)

    Y = data[:,p['fea_num']:p['fea_num']+p['out_num']]
    #train model
    if p['kfold']:
        cvr, cvscores, r2, predicted, gtest, model, history = train_kfold(X,Y,p)
    else:
        cvr, cvscores, r2, predicted, gtest, model, history = train_all(X,Y,p)

    #save current model
    model.save(p['model_path']+p['model_name'])
    model.save_weights(p['model_path']+'weights.h5')

    logging.info('Mean MSE %: ' + str(np.mean(cvscores)) + '; MSE std: ' + str(np.std(cvscores)))
    logging.info('Mean Correlation %: ' + str(np.mean(cvr)) + '; Correlation std: ' + str(np.std(cvr)))
    logging.info('R-Squared %: ' + str(np.mean(r2)) + '; Correlation std: ' + str(np.std(r2)))

    #record history to list
    import csv
    from itertools import izip
    with open('train_history.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(izip(history.history['loss'], history.history['val_loss']))

    #plot to check overfitting, result correlation
    if p['plot_on']:
        vis_train(history, predicted, gtest)

    os.system('espeak "done"')
    logging.info('Training Completed')

if __name__ == '__main__':
    logging.info("Training Time (s): " + str(timeit.timeit("main()", setup="from __main__ import main", number=1)))
