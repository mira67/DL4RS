# DL4RS
Deep Learning in Remote Sensing Applications

[MPFprj]
Train/Test Pipeline for Melt Pond Prediction with NN Models

[Prototype]
A simple workflow for basic NN model testings
  [Key Structure]
    train.py - load train/validate/test data
                  - train/evaluate/save model
    test.py - load test data and record results to csv and sql database
    config.ini - data path, model, sqls configuration file, used with train.py, test.py


pre-trained model: good model trained for melt pond fraction prediction, 7-10-10 tanh, 0.2 dropout 10 units layer, adam optimizer, month 5,6,7, trained for 500 times, month 8 trained for 10000, well can reduce iterations, 10000 is just a fun trial


