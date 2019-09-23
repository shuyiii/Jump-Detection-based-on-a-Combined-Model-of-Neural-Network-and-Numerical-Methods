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

##############################################################################################################
temp=loadmat('/users/PAS1263/osu8085/1D_jump_generate_near_bound/random_f_common_method/train_data.mat', mdict=None, appendmat=True)
temp=temp['train_data']
x_train=[(obs-np.mean(obs))/np.std(obs) for obs in temp]
x_train=np.asarray(x_train).reshape(1000000,1,202,1)
y_train=loadmat('/users/PAS1263/osu8085/1D_jump_generate_near_bound/random_f_common_method/J_train_loss2.mat', mdict=None, appendmat=True)
y_train=y_train['J_train_loss2']
#######################################################################
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
model.add(Dense(201))

filename="./model_weights_kernel2_norm_near_100w_no_res.hdf5"
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=0, save_best_only=True,mode='max')
callbacks_list = [checkpoint]
adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer = "adam", loss = root_mean_squared_error, 
              metrics =["accuracy"])
model.fit(x_train, y_train,validation_split=0.1, batch_size=5000, epochs=300, callbacks=callbacks_list,verbose=0)
