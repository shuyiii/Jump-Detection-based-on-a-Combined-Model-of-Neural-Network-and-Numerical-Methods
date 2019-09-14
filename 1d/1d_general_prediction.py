"""
@author: Shuyi Wang
general 1d testing for functions with different domain, grid size and length.
"""
import tensorflow as tf
import keras
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Sequential
from keras.layers import Dense,Flatten,Activation,Convolution2D
from keras.optimizers import SGD, Adam
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import *
##########load model ##########################################################
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

filepath="/1d/"
model.load_weights(filepath+'model_weights_kernel2.h5')
###############################################################################
def change_data(data):#if length of function is less than 202, we need to extend it to 202
    if len(data)<202:
        a=(202-(len(data)))//2
        b=202-len(data)-a
        pre=[data[0]]*a
        post=[data[-1]]*b
        temp=np.concatenate((pre, data), axis=0)
        temp=np.concatenate((temp,post), axis=0)
    else:
        temp=data
    return temp
###############################################################################
def norm(data):#normalize each piece of function, in this way either 1 data or multiple data can be processed together.
    re=[]
    for i in range(len(data)):
        obs=data[i]
        temp=(obs-np.mean(obs))/np.std(obs)
        re.append(temp.reshape(1,202,1))
    return np.asarray(re)
###############################################################################
def predict(data,m):#m is the sliding window, when there is no enough length, the final testing piece has the start point at last 202 place.
    n=len(data)
    data=change_data(data)
    if n>202:  
       result=[0]*n
    else:
       result=[0]*202
    ind=0
    while ind+202<=n:
          temp_result=model.predict(norm(data[ind:ind+202].reshape(1,202)))
          result[ind:ind+201]=[max(result[ind+k],temp_result[0,k]) for k in range(201)]
          ind+=m
    if ind<n:
          temp_result=model.predict(norm(data[max(n,202)-202:].reshape(1,202)))
          result[max(n,202)-202:]=[max(result[max(n,202)-202+k],temp_result[0,k]) for k in range(201)]
    if n>=202:
        return result
    else:
        return result[(202-n)//2:(202-n)//2+n-1]
###############################################################################
def predict_loc(result,threshold):#give a threshold, predict the location of jump 
    return np.where(result>threshold)
###############################################################################
def plot_function(result,data,test_location,name):
    n=len(data)
    x0=np.linspace(-1,1,n)
    p1,=plt.plot(x0,data,'.')#original function 
    result.append(0)
    p2,=plt.plot(x0,result)
    loc=np.where(test_location!=-1)[0]
    for k in loc:
        plt.axvline(test_location[k])
    plt.legend([p1,p2], ['original', 'cnn'])
    plt.title(str(i))
    plt.show()
    plt.savefig('/1d/1d_general_prediction'+name+'.svg')
    plt.close()
###############################################################################    
    
    
    
##############an example#######################################################
path='D:/paper_material/1d/1d_general_prediction/data_example/'
temp=loadmat(path+'[-1,1]1002.mat', mdict=None, appendmat=True)
test_data=temp['output_function']
location=temp['Jlocation']
n=1002
for i in range(30):
    data=test_data[i]
    result=predict(data,100)
    result.append(0)
    x0=np.linspace(-1,1,n)
    p1,=plt.plot(x0,data,'.',markersize=1)
    p2,=plt.plot(x0,result,linewidth=0.5)
    for d in location[i]:
        if d!=99:
           plt.axvline((d+1)//(2/(n-1))*(2/(n-1))-1,linewidth=0.5)#because 0 is the first point
    plt.legend([p1,p2], ['original data', 'CNN output'])
    plt.title('function on [-1,1] with data length=1002')
    plt.show()
    plt.savefig('/1d/1d_general_prediction/plot/'+'function_length_1002_'+str(i+1)+'.svg')
    plt.close()
##############another example##################################################
a=[32,52,72,92,102,202,302,402,502,602,702,802,902,1002]
temp=loadmat(path+'data1.mat', mdict=None, appendmat=True)
location=temp['Jlocation']
temp=temp['f_out']
for i in range(14):
    n=a[i]
    data=temp[i][:n]
    result=predict(data,100)
    result.append(0)
    x0=np.linspace(-1,1,n)
    p1,=plt.plot(x0,data,'.',markersize=3)
    p2,=plt.plot(x0,result,linewidth=1)
    for d in location[0]:
        if d!=99:
           plt.axvline((d+1)//(2/(n-1))*(2/(n-1))-1,linewidth=0.5,color='green')#because 0 is the first point
    plt.legend([p1,p2], ['original data', 'CNN output'])
    plt.title('('+chr(i+96)+')')
    plt.show()
    plt.savefig('/1d/1d_general_prediction/plot/'+'function_length_'+str(n)+'.svg')
    plt.close()


