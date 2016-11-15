"""
Test Main
Author: Qi Liu
"""
import os
import logging
import timeit
import ConfigParser
import numpy as np
from prjutil import read_config
from keras.models import load_model
from dbaccess import pdread,sqlwrite
import pandas as pd
from sklearn.preprocessing import normalize

def model_test(p):
    #load model
    model = load_model(p['model_path']+p['model_name'])
    df = pdread(p['test_sql'])
    df = df.replace(-9999, 0)
    #df['julian'][(df['julian'] < 213) & (df['julian'] > 152)] = 0.1
    basen = 2
    julian_m = 197.5
    alpha = 10
    beta = 0.01

    #df['julian'] = alpha*np.power(df['julian']-julian_m, basen)/np.power(274,basen)+beta

    print df.head(n=5)
    #df = minmaxscaler(df)
    data = df.as_matrix()
    X_predict = data[:,0:]
    #X_predict = normalize(X_predict,norm='l2',axis=0)
    # print X_predict[1:5,:]
    #predict with model
    Y_predict = model.predict(X_predict)*100
    df = pd.DataFrame(Y_predict, columns=['MPF', 'IF', 'WF'])
    #record to mysql
    df.to_csv(p['result_path']+p['test_result_csv'], sep=',', encoding='utf-8')
    #also write to database
    sqlwrite(p['result_path'], p['test_result_csv'], p['csvtosql'])

def dual_model(p):
    #load model
    model = load_model(p['model_path']+p['model_name'])
    #model2 = load_model(p['model_path']+p['model_name2'])

    df = pdread(p['test_sql'])
    df = df.replace(-9999, 0)
    #print df.head(n=5)

    data = df.as_matrix()
    attr_n = 7
    attr = data[:,0:attr_n]#year,month,day,nrow,ncol
    X_predict = data[:,attr_n:attr_n+p['fea_num']]
    d_month = data[:,p['fea_num']]

    #X_predict = normalize(X_predict,norm='l2',axis=0)
    print X_predict[1:attr_n,:]

    Y_predict = model.predict(X_predict)*100

    final_data = np.concatenate((attr,Y_predict), axis=1)

    print final_data.shape

    df = pd.DataFrame(final_data, columns=['year','month','day','nrow','ncol','qc','cloud','MPF', 'IF', 'WF'])
    #record to mysql
    with open(p['result_path']+p['test_result_csv'], 'a') as f:
        df.to_csv(f, sep=',', encoding='utf-8',header=False)
    #also write to database
    sqlwrite(p['result_path'], p['test_result_csv'], p['csvtosql'])

def main():
    logging.basicConfig(filename='testing.log', level=logging.INFO)
    logging.info('Started Testing')
    #read config
    p = read_config();
    logging.info('Testing with Model: ' + str(p['model_id']))
    #model_test(p)
    dual_model(p)
    #sqlwrite(p['result_path'], p['test_result_csv'], p['csvtosql'])
    os.system('espeak "done"')

if __name__ == '__main__':
    logging.info("Testing Time (s): " + str(timeit.timeit("main()", setup="from __main__ import main", number=1)))
