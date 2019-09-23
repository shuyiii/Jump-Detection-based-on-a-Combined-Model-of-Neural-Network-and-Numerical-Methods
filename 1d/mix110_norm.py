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
import pylab as pl
from keras.models import Model
from keras.callbacks import ModelCheckpoint

#########################################################################################################
temp=loadmat('/users/PAS1263/osu8085/0.25_1_mil/train_data.mat', mdict=None, appendmat=True)
temp=temp['train_data']
x_train=[(obs-np.mean(obs))/np.std(obs) for obs in temp]
x_train=np.asarray(x_train).reshape(1000000,1,202,1)
y_train=loadmat('/users/PAS1263/osu8085/0.25_1_mil/J_train_loss_both_JC.mat', mdict=None, appendmat=True)
y_train=y_train['J_train_loss_both_JC']
X_train=loadmat('/users/PAS1263/osu8085/0.25_1_mil/train_data.mat', mdict=None, appendmat=True)
X_train=X_train['train_data']
train_C_location=loadmat('/users/PAS1263/osu8085/0.25_1_mil/train_C_location.mat', mdict=None, appendmat=True)
train_C_location=train_C_location['train_C_location']
train_J_location=loadmat('/users/PAS1263/osu8085/0.25_1_mil/train_J_location.mat', mdict=None, appendmat=True)
train_J_location=train_J_location['train_J_location']
x_train=x_train.astype(np.float64)
y_train=y_train.astype(np.float64)
##########################################################################################################
x_test=loadmat('/users/PAS1263/osu8085/0.25_1_mil/test_data.mat', mdict=None, appendmat=True)
x_test=x_test['test_data']
x_test=x_test.reshape(10000,1,202,1)
y_test=loadmat('/users/PAS1263/osu8085/0.25_1_mil/J_test_loss_both_JC.mat', mdict=None, appendmat=True)
y_test=y_test['J_test_loss_both_JC']
X_test=loadmat('/users/PAS1263/osu8085/0.25_1_mil/test_data.mat', mdict=None, appendmat=True)
X_test=X_test['test_data']
test_C_location=loadmat('/users/PAS1263/osu8085/0.25_1_mil/test_C_location.mat', mdict=None, appendmat=True)
test_C_location=test_C_location['test_C_location']
test_J_location=loadmat('/users/PAS1263/osu8085/0.25_1_mil/test_J_location.mat', mdict=None, appendmat=True)
test_J_location=test_J_location['test_J_location']
x_test=x_test.astype(np.float64)
y_test=y_test.astype(np.float64)
########################################################################################################
#swich to -1 and 1
for i in range(1000000):
    for j in range(201):
        if y_train[i,j]==2:
            y_train[i,j]=10           

for i in range(10000):
    for j in range(201):
        if y_test[i,j]==2:
            y_test[i,j]=10
########################################################################################################
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


filename="/users/PAS1263/osu8085/0.25_1_mil/ker2_1_and_10_normalize.h5"
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=0, save_best_only=True,mode='max')
callbacks_list = [checkpoint]
adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer = "adam", loss = root_mean_squared_error, 
              metrics =["accuracy"])
model.fit(x_train, y_train,validation_split=0.1, batch_size=20000, epochs=300, callbacks=callbacks_list,verbose=0)
