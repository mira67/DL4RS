"""
Train Main
Author: Qi Liu
Date: Oct.27.2016
"""
import os
import logging
import timeit
import ConfigParser
from prjutil import read_config
from dbaccess import pdread
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
import numpy as np
from mpfmodel import train_model

def train_kfold(n_splits,X,Y,p):
    logging.info('kfold training...')
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    cvscores = []
    cvr = []
    for train, test in kfold.split(X):
        predicted, gtest, model = train_model(X[train],Y[train],X[test],Y[test],p)
        score = model.evaluate(X[test], Y[test], batch_size=p['bsize'], verbose=0)
        mpf_mse = np.absolute(np.mean(predicted[:,0],axis=0)-np.mean(gtest[:,0],axis=0))/np.mean(gtest[:,0],axis=0)*100
        r = np.corrcoef(predicted[:,0], gtest[:,0])[1,0]
        cvr.append(r)
        cvscores.append(mpf_mse)
    return cvr,cvscores,model

def train_all(X,Y,p):
    np.random.seed(1)
    msk = np.random.rand(len(df)) < 0.8
    X_train = X[msk]
    X_test = X[~msk]
    Y_train = Y[msk]
    Y_test = Y[~msk]
    predicted, gtest, model = train_model(X_train,Y_train,X_test,Y_test,p)
    score = model.evaluate(X_test, Y_test, batch_size=p['bsize'], verbose=0)
    mpf_mse = np.absolute(np.mean(predicted[:,0],axis=0)-np.mean(gtest[:,0],axis=0))/np.mean(gtest[:,0],axis=0)*100
    r = np.corrcoef(predicted[:,0], gtest[:,0])[1,0]
    cvr.append(r)
    cvscores.append(mpf_mse)
    #save current model
    model.save('recent_model_m3.h5')
    return cvr, cvscores, model

def main():
    logging.basicConfig(filename='training.log', level=logging.INFO)
    logging.info('Started Training')
    #read config
    p = read_config();
    logging.info('Training with Model: ' + str(p['model_id']))
    #get train/validation/test data
    df = pdread(p['train_sql'])
    df = df.replace(-9999, 0)
    data = df.as_matrix()
    X = data[:,0:p['fea_num']]
    Y = data[:,p['fea_num']:p['fea_num']+p['out_num']]
    #train model
    if p['kfold']:
        cvr, cvscores, model = train_kfold(n_splits,X,Y,p)
    else:
        cvr, cvscores, model = train_all(X,Y,p)

    #save current model
    model.save(p['model_path']+p['model_name'])

    logging.info('Mean MSE %: ' + str(np.mean(cvscores)) + '; MSE std: ' + str(np.std(cvscores)))
    logging.info('Mean Correlation %: ' + str(np.mean(cvr)) + '; Correlation std: ' + str(np.std(cvr)))

    os.system('espeak "Congratulations, Your Training is done"')
    logging.info('Training Completed')

if __name__ == '__main__':
    logging.info("Training Time (s): " + str(timeit.timeit("main()", setup="from __main__ import main", number=1)))
