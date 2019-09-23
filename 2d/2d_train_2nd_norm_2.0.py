# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:55:23 2019

@author: wang7
"""
#create new training data to piece of 4*4 for training
import keras
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Sequential
from keras.layers import MaxPooling2D,Lambda,Dense,BatchNormalization,Flatten,Reshape,Dropout,Activation,Convolution2D,GlobalAveragePooling1D
from keras.optimizers import SGD, Adam
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import *
import pylab as pl
from keras.models import Model
from sklearn import preprocessing
import tensorflow as tf
from matplotlib import colors
import matplotlib.patches as patches
from keras.callbacks import ModelCheckpoint
###############################################################################
def root_mean_squared_error(y_pred,y_true):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

model = Sequential()
model.add(Convolution2D(32, (4,4),strides=(1,1),input_shape=(101,101,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
model.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
model.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100))

adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer = "adam", loss = root_mean_squared_error, 
              metrics =["accuracy"])

filepath="/2D_models/new_train/"
model.load_weights(filepath+'model_weights_10_10_normalization_each.h5')
###############################################################################
num=76045
folder='/data/circle_1mil/'
train_data=[]
train_label=[]
t=np.load('label_10_circle.npy')
total_circle=0#record how many samples of circle used 
total_true_circle=0#record how many samples contain jumps
for i in range(1,num+1):
    temp=loadmat(folder+'image'+str(i)+'.mat')
    data=temp['image']
    data=(data-np.mean(data))/np.std(data)
    label=temp['label1']
    temp_label=t[i-1]
    pred_label=model.predict(data.reshape(1,101,101,1))>0.1
    for j in range(10):
        for k in range(10):
            if pred_label[j*10+k]==1:
               total_circle+=1
               if temp_label[j*10+k]==1:
                   total_true_circle+=1
               tmp=data[10*j:10*(j+1)+1,10*k:10*(k+1)+1]
               tmp=(tmp-np.mean(tmp))/np.std(tmp)
               train_data.append(tmp.reshape(11,11,1))
               train_label.append(label[10*j:10*(j+1),10*k:10*(k+1)].ravel())
num=68076
folder='/data/line_curve/'
t=np.load('label_10_line.npy')
total_line=0#record how many samples of line used 
total_true_line=0#record how many samples contain jumps
for i in range(1,num+1):
    temp=loadmat(folder+'image'+str(i)+'.mat')
    data=temp['image']
    data=(data-np.mean(data))/np.std(data)
    label=temp['label1']
    temp_label=t[i-1]
    pred_label=model.predict(data.reshape(1,101,101,1))>0.1
    for j in range(10):
        for k in range(10):
            if pred_label[j*10+k]==1:
               total_line+=1
               if temp_label[j*10+k]==1:
                  total_true_circle+=1
               tmp=data[10*j:10*(j+1)+1,10*k:10*(k+1)+1]
               tmp=(tmp-np.mean(tmp))/np.std(tmp)
               train_data.append(tmp.reshape(11,11,1))
               train_label.append(label[10*j:10*(j+1),10*k:10*(k+1)].ravel())


train_data=np.asarray(train_data)
train_label=np.asarray(train_label)
###############################################################################################
def root_mean_squared_error(y_pred,y_true):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

model = Sequential()
model.add(Convolution2D(32, (2,2),strides=(1,1),input_shape=(11,11,1),activation='relu'))
model.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
model.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
model.add(Flatten())
model.add(Dropout(0.1))
model.add(Dense(100))

adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer = "adam", loss = root_mean_squared_error, 
              metrics =["accuracy"])

#filepath="/2D_models/new_train/"
#model.load_weights(filepath+'model_weights_10_10_local_normalize.h5')
filename="/2D_models/new_train/model_weights_10_10_local_normalize_2.0.hdf5"
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=0, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(train_data,train_label, validation_split=0.1, callbacks=callbacks_list,batch_size=2000, epochs=50, verbose=0)
#model.fit(train_data,train_label, batch_size=2000, epochs=50, verbose=0)
print(total_circle)
print(total_true_circle)
print(total_line)
print(total_true_line)
#get each potential coordination into predicting (for testing)
#save its index (each one)
 
