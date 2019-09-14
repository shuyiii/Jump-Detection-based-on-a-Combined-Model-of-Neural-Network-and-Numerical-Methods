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

num=76045
folder='/fs/project/PAS1263/data/circle_1mil/'
train_data=[]
train_label=[]
for i in range(1,num+1):
    temp=loadmat(folder+'image'+str(i)+'.mat')
    img=temp['image'] 
    img=(img-np.mean(img))/np.std(img)
    train_data.append(img.reshape(101,101,1))
    train_label.append(temp['label1'].ravel())
num=68076
folder='/fs/project/PAS1263/data/line_curve/'
for i in range(1,num+1):
    temp=loadmat(folder+'image'+str(i)+'.mat')
    img=temp['image'] 
    img=(img-np.mean(img))/np.std(img)
    train_data.append(img.reshape(101,101,1))
    train_label.append(temp['label1'].ravel())

train_data=np.asarray(train_data)
train_label=np.asarray(train_label)
###############################################################################################
def root_mean_squared_error(y_pred,y_true):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

model = Sequential()
model.add(Convolution2D(32, (4,4),strides=(1,1),input_shape=(101,101,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
model.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10000))

adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer = "adam", loss = root_mean_squared_error, 
              metrics =["accuracy"])

filepath="/users/PAS1263/osu8085/2D_models/new_train/"
model.load_weights(filepath+'model_weights_100_100_normalization_each.h5')
history = model.fit(train_data,train_label, batch_size=1000, epochs=30, verbose=0)

model.save_weights(filepath+'model_weights_100_100_normalization_each.h5')
