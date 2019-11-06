import tensorflow as tf
import keras
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Sequential
from keras.layers import MaxPooling2D,Lambda,Dense,BatchNormalization,Flatten,Reshape,Dropout,Activation,Convolution2D,GlobalAveragePooling1D
from keras.optimizers import SGD, Adam
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from keras.models import Model
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

filepath="/users/PAS1263/osu8085/2D_models/new_train/"
model.load_weights(filepath+'model_weights_10_10_normalization_each.h5')
###############################################################################
num=76045
folder='/fs/project/PAS1263/data/circle_1mil/'
train_data=[]
train_label=[]
t=np.load('label_10_circle.npy')
for i in range(1,num+1):
    temp=loadmat(folder+'image'+str(i)+'.mat')
    data=temp['image']
    data=(data-np.mean(data))/np.std(data)
    label=temp['label1']
    temp_label=t[i-1]
    pred_label=model.predict(data.reshape(1,101,101,1))>=0
    for j in range(10):
        for k in range(10):
               tmp=data[10*j:10*(j+1)+1,10*k:10*(k+1)+1]
               tmp=(tmp-np.mean(tmp))/np.std(tmp)
               train_data.append(tmp.reshape(11,11,1))
               train_label.append(label[10*j:10*(j+1),10*k:10*(k+1)].ravel())
num=68076
folder='/fs/project/PAS1263/data/line_curve/'
t=np.load('label_10_line.npy')
for i in range(1,num+1):
    temp=loadmat(folder+'image'+str(i)+'.mat')
    data=temp['image']
    data=(data-np.mean(data))/np.std(data)
    label=temp['label1']
    temp_label=t[i-1]
    for j in range(10):
        for k in range(10):
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

filename=filepath+"model_weights_10_10_local_normalize_3.0.hdf5"
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(train_data,train_label, validation_split=0.1, callbacks=callbacks_list,batch_size=5000, epochs=500, verbose=0)