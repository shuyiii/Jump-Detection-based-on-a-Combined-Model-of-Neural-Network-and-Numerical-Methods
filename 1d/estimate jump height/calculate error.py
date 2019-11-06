"""
@author: wang7
"""
import tensorflow as tf
import keras
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Sequential
from keras.layers import Lambda,Dense,BatchNormalization,Flatten,Reshape,Dropout, Activation,Convolution2D,GlobalAveragePooling1D
from keras.optimizers import SGD, Adam
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import *
import sys
from keras.models import Model
##########load model ###############################################################
def root_mean_squared_error(y_pred,y_true):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 
                  
model = Sequential()
model.add(Convolution2D(24, (1,2),strides=(1,1),input_shape=(1,202,1)))
model.add(Activation('relu'))
model.add(Convolution2D(24, (1,2),strides=(1,1),input_shape=(1,202,1)))
model.add(Activation('relu'))
model.add(Convolution2D(24, (1,2),strides=(1,1),input_shape=(1,202,1)))
model.add(Activation('relu'))
model.add(Convolution2D(24, (1,2),strides=(1,1),input_shape=(1,202,1)))
model.add(Activation('relu'))
model.add(Convolution2D(24, (1,2),strides=(1,2),input_shape=(1,202,1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(201))#relu or not?  should restrict each to 0 and 1
model.load_weights('/users/PAS1263/osu8085/generate_data_with_height/model_with_real_height.h5')
###############################################################################
temp=loadmat('/users/PAS1263/osu8085/generate_data_with_height/test_data.mat', mdict=None, appendmat=True)
temp=temp['test_data']
x_test=[(obs-np.mean(obs))/np.std(obs) for obs in temp]
x_test=np.asarray(x_test).reshape(10000,1,202,1)
y_test=loadmat('/users/PAS1263/osu8085/generate_data_with_height/J_test_loss2.mat', mdict=None, appendmat=True)
y_test=y_test['J_test_loss2']
X_test=loadmat('/users/PAS1263/osu8085/generate_data_with_height/test_data.mat', mdict=None, appendmat=True)
X_test=X_test['test_data']
test_location=loadmat('/users/PAS1263/osu8085/generate_data_with_height/test_location.mat', mdict=None, appendmat=True)
test_location=test_location['test_location']
y_test=np.asarray(y_test)
###############################################################################
#calculate original mean squared error, only about jump location
result=model.predict(x_test)
scaled_error=0
error=0
count=0
for i in range(10000):
    for j in range(3):
        if test_location[i,j]!=-1:
           scaled_error+=(result[i][test_location[i,j]]-y_test[i][test_location[i,j]])**2
           error+=((result[i][test_location[i,j]]-y_test[i][test_location[i,j]])*np.std(X_test[i])+np.mean(X_test[i]))**2
           count+=1
scaled_error=scaled_error/count
error=error/count
print(scaled_error)
print(error)
