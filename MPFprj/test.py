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

def model_test(p):
    #load model
    model = load_model(p['model_path']+p['model_name'])
    df = pdread(p['test_sql'])
    df = df.replace(-9999, 0)
    #df = minmaxscaler(df)
    data = df.as_matrix()
    X_predict = data[:,0:]
    #predict with model
    Y_predict = model.predict(X_predict)*100
    df = pd.DataFrame(Y_predict, columns=['MPF', 'IF', 'WF'])
    #record to mysql
    df.to_csv(p['result_path']+p['test_result_csv'], sep=',', encoding='utf-8')
    #also write to database
    sqlwrite(p['result_path'], p['test_result_csv'], p['csvtosql'])

def main():
    logging.basicConfig(filename='testing.log', level=logging.INFO)
    logging.info('Started Testing')
    #read config
    p = read_config();
    logging.info('Testing with Model: ' + str(p['model_id']))
    model_test(p)
    os.system('espeak "done"')

if __name__ == '__main__':
    logging.info("Testing Time (s): " + str(timeit.timeit("main()", setup="from __main__ import main", number=1)))
