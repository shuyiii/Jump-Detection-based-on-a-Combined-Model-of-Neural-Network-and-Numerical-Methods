import tensorflow as tf
import keras
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Sequential
from keras.layers import Lambda,Dense,BatchNormalization,Flatten,Reshape,Dropout, Activation,Convolution2D,MaxPooling2D
from keras.optimizers import SGD, Adam
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import *
from keras.models import Model
import math
####################################################################################
x_test=loadmat('E:/CNN/detect discontinuity/1D data/1D data/0-3jump near bound/test_data.mat', mdict=None, appendmat=True)
x_test=x_test['test_data']
x_test=x_test.reshape(10000,1,202,1)
X_test=loadmat('E:/CNN/detect discontinuity/1D data/1D data/0-3jump near bound/test_data.mat', mdict=None, appendmat=True)
X_test=X_test['test_data']
test_location=loadmat('E:/CNN/detect discontinuity/1D data/1D data/0-3jump near bound/test_location.mat', mdict=None, appendmat=True)
test_location=test_location['test_location']
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
model.load_weights('D:/paper_material/1d/model_weights_kernel2.h5')
###############################################################################
threshold=0
total=np.sum(test_location!=-1)#number of jumps in all test data
result=model.predict(x_test)
minimum=1
for i in range(1,10001):
    thresh=i*0.0001 
    num=np.sum(result>thresh)
    if abs(num/total-1)<minimum:
        minimum=abs(num/total-1)
        threshold=thresh
print(threshold)
print(minimum)
#0.9712000000000001
#0.0
##############################################################################
