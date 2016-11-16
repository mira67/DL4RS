"""
Analyze importance of input feature
Good for simple model

Author: Qi Liu, 11/2016
"""
import os
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

mpath = "/home/mirabot/googledrive/deeplearning/seaiceprj/mpfdata/model/"
mname = "igarss1110-m3-2.h5"
model = load_model(mpath+mname)

font_size = 11

plt.style.use(u'seaborn-paper')
plt.rcParams['axes.labelsize'] = font_size

fig, ax = plt.subplots(1, 1, num=1)

w = model.layers[0].get_weights()[0]
w = w/np.amax(abs(w))
#w_min, w_max = -np.abs(w).max(), np.abs(w).max()
img = ax.imshow(w, interpolation='none',aspect='auto',cmap=plt.get_cmap('coolwarm'))
#plt.pcolor(w,cmap='RdBu',vmin=w_min, vmax=w_max)
labels = [0,1,2,3,4,5,6,7]
ax.set_xticklabels(labels,ha='center', minor=False,fontsize = 12)
ax.set_yticklabels(labels,ha='center', minor=False,fontsize = 12)
plt.xlabel('Unit Index')
plt.ylabel('MODIS Band Index')
plt.colorbar(img)
ax.tick_params(labelsize=font_size)
plt.show()

"""calculate input relative importance"""
w1 = model.layers[0].get_weights()[0]
w2 = model.layers[3].get_weights()[0]
w3 = model.layers[6].get_weights()[0]
w4 = model.layers[9].get_weights()[0]
w5 = model.layers[12].get_weights()[0]

mvp = abs(w1).dot(abs(w2)).dot(abs(w3)).dot(abs(w4)).dot(abs(w5))
#print(mvp)
mvpr = mvp.sum(axis=1)
print(mvpr)

for r in mvpr:
    print(r/mvpr.sum()*100)
