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
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

num=76045
folder='/fs/project/PAS1263/data/circle_1mil/'
train_data=[]
for i in range(1,num+1):
    temp=loadmat(folder+'image'+str(i)+'.mat')
    img=temp['image']
    img=(img-np.mean(img))/np.std(img)
    train_data.append(img.reshape(101,101,1))
label_10_circle=np.load('label_10_circle.npy')
num=68076
folder='/fs/project/PAS1263/data/line_curve/'
for i in range(1,num+1):
    temp=loadmat(folder+'image'+str(i)+'.mat')
    img=temp['image']
    img=(img-np.mean(img))/np.std(img)
    train_data.append(img.reshape(101,101,1))
label_10_line=np.load('label_10_line.npy')
train_label=np.concatenate((label_10_circle,label_10_line))

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
model.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100))

adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer = "adam", loss = root_mean_squared_error, metrics =["accuracy"])
filepath="/users/PAS1263/osu8085/2D_models/new_train/"
filename=filepath+"/model_weights_10_10_normalization_each.h5"
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit(train_data,train_label, validation_split=0.1, callbacks=callbacks_list,batch_size=1000, epochs=500, verbose=0)
#print(np.mean(train_data))
#print(np.std(train_data))
