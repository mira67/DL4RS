"""
Visualization Module
Author: Qi Liu
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
plt.style.use(u'seaborn-paper')

def vis_train(history,predicted,gt):
    fs = 14
    #train/validation cost curve
    plt.figure(1)
    gs = GridSpec(2, 2)
    plt.subplot(gs[0, :])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss',fontsize=fs, color='black')
    plt.xlabel('epoch',fontsize=fs, color='black')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.tick_params(axis='both', which='major', labelsize=fs)

    #predicted vs ground truth correlation
    t = np.arange(0, len(gt), 1)
    plt.subplot(gs[1, 0])
    plt.scatter(predicted[:,0], gt[:,0])
    axes = plt.gca()
    m, b = np.polyfit(predicted[:,0], gt[:,0], 1)
    X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
    r = np.corrcoef(predicted[:,0], gt[:,0])[1,0]
    plt.plot(X_plot, m*X_plot + b, 'r-',label="r = %2.2f"%(r))
    plt.legend(loc=2)
    plt.ylabel('GT MPF(%)',fontsize=fs, color='black')
    plt.xlabel('Predicted MPF(%)',fontsize=fs, color='black')
    plt.tick_params(axis='both', which='major', labelsize=fs)

    plt.subplot(gs[1, 1])
    plt.plot(t, gt[:,0], 'b-', label='Ground Truth')
    plt.plot(t, predicted[:,0], 'r-',label='Predicted')
    plt.ylabel('MPF(%)',fontsize=fs, color='black')
    plt.xlabel('Grid Index, Not ordered',fontsize=fs, color='black')
    plt.tick_params(axis='both', which='major', labelsize=fs)
    plt.legend(loc=2)

    plt.show()

def vis_test(history,predicted,gt):
