from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import keras
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Sequential
from keras.layers import Conv3D, MaxPool3D,Lambda,Dense,BatchNormalization,Flatten,Reshape,Dropout,Activation
from keras.optimizers import SGD, Adam
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import *
import sys
import pylab as pl
from keras.models import Model
from sklearn import preprocessing
import cv2
import os
import tensorflow as tf
import skimage
from matplotlib import colors
import matplotlib.patches as patches
#############################################################################################################
folder='D:/sphere_cut/'
test_data=[]
test_label=[]
for i in range(5000,5021):    
    temp=loadmat(folder+'image'+str(i)+'.mat')
    tmp=temp['f']
    tmp=(tmp-np.mean(tmp))/np.std(tmp)
    test_data.append(tmp.reshape(101,101,101,1))#K.image_dim_ordering()='tf'
    test_label.append(temp['label1'])

test_data=np.asarray(test_data)
test_label=np.asarray(test_label)
#############################################################################################################
model = Sequential()
model.add(Conv3D(12, (5,5,5),strides=(5,5,5),input_shape=(101,101,101,1),activation='relu'))
model.add(MaxPool3D(pool_size=(2,2,2)))
model.add(Conv3D(6, (2,2,2),strides=(2,2,2),activation='relu'))
model.add(Flatten())
model.add(Dense(1000))
model.load_weights('D:/paper_material/3d/model_weights_3D_10_10.h5')
result=model.predict(test_data)
###########################################################################################################
threshold=0.4
for i in range(10):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    true_loc=np.where(test_label[i]==1)#should be array
    ax.scatter3D(true_loc[2]+0.5, true_loc[1]+0.5, true_loc[0]+0.5, c=true_loc[0]+0.5)#middle point
    x,y,z=[],[],[]
    pred_loc=np.where(result[i]>threshold)[0]
    for loc in pred_loc:
        z.append(loc//100*10)
        x.append(loc%100%10*10)
        y.append(loc%100//10*10)
    for j in range(len(x)):
        ax.plot3D([x[j],x[j]+10], [y[j],y[j]], [z[j],z[j]], 'red')
        ax.plot3D([x[j],x[j]+10], [y[j]+10,y[j]+10], [z[j],z[j]], 'red')
        ax.plot3D([x[j],x[j]+10], [y[j],y[j]], [z[j]+10,z[j]+10], 'red')
        ax.plot3D([x[j],x[j]+10], [y[j]+10,y[j]+10], [z[j]+10,z[j]+10], 'red')
        ax.plot3D([x[j],x[j]], [y[j],y[j]+10], [z[j],z[j]], 'red')
        ax.plot3D([x[j]+10,x[j]+10], [y[j],y[j]+10], [z[j],z[j]], 'red')
        ax.plot3D([x[j],x[j]], [y[j],y[j]+10], [z[j]+10,z[j]+10], 'red')
        ax.plot3D([x[j]+10,x[j]+10], [y[j],y[j]+10], [z[j]+10,z[j]+10], 'red')
        ax.plot3D([x[j],x[j]], [y[j],y[j]], [z[j],z[j]+10], 'red')
        ax.plot3D([x[j]+10,x[j]+10], [y[j],y[j]], [z[j],z[j]+10], 'red')
        ax.plot3D([x[j],x[j]], [y[j]+10,y[j]+10], [z[j],z[j]+10], 'red')
        ax.plot3D([x[j]+10,x[j]+10], [y[j]+10,y[j]+10], [z[j],z[j]+10], 'red')
    plt.show()
    plt.savefig('D:/paper_material/3d/plot/'+str(i)+'.svg')
    plt.close()
