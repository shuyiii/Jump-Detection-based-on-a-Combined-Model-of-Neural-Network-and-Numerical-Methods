# -*- coding: utf-8 -*-
"""
@author: wang7
2d general detection CNN+CNN
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.io as sio
import math
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Sequential
from keras.layers import MaxPooling2D,Lambda,Dense,BatchNormalization,Flatten,Reshape,Dropout,Activation,Convolution2D,GlobalAveragePooling1D
from keras.optimizers import SGD, Adam
from scipy.io import loadmat
from keras.models import Model
from sklearn import preprocessing
from matplotlib import colors
######load model###############################################################
def root_mean_squared_error(y_pred,y_true):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

model = Sequential()
model.add(Convolution2D(32, (4,4),strides=(1,1),input_shape=(101,101,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
model.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
model.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
model.add(Flatten())
model.add(Dense(100))

model.load_weights('/2d/2stepmodelupdate/model_weights_10_10_normalize.h5')


model2 = Sequential()
model2.add(Convolution2D(32, (2,2),strides=(1,1),input_shape=(11,11,1),activation='relu'))
model2.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
model2.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
model2.add(Flatten())
model2.add(Dense(100))

model2.load_weights('/2d/2stepmodelupdate/model_weights_10_10_local_normalize.h5')
###############################################################################
def extend_function(data):
    if len(data)>=101 and len(data[0])>=101:
        return data
    if len(data)<101 and len(data[0])<101:
       a=(101-len(data))//2
       b=101-len(data)-a
       c=(101-len(data[0]))//2
       d=101-len(data[0])-c
       temp_data=np.zeros([101,101])#extend 10 grids for top, bottom, left, right
       temp_data[0:a,0:c]=data[0,0] #top left
       temp_data[0:a,101-d:]=data[0,-1] #top right
       temp_data[101-b:,0:c]=data[-1,0] #bottom left
       temp_data[101-b:,101-d:]=data[-1,-1] #bottom left
       for i in range(a,101-b):
             temp_data[i,0:c]=data[i-a,0]#left
             temp_data[i,101-d:]=data[i-a,-1]#right
       for i in range(c,101-d):
             temp_data[0:a,i]=data[0,i-c]#top     
             temp_data[101-b:,i]=data[-1,i-c]#bottom 
       temp_data[a:101-b,c:101-d]=data
       return temp_data
    if len(data)<101:
       a=(101-len(data))//2
       b=101-len(data)-a
       temp_data=np.zeros([101,len(data[0])])
       for i in range(len(data[0])):
           temp_data[0:a,i]=data[0,i]
           temp_data[101-b:,i]=data[-1,i]
       temp_data[a:101-b,:]=data
       return temp_data
    if len(data[0])<101:
       c=(101-len(data[0]))//2
       d=101-len(data[0])-c
       temp_data=np.zeros([len(data),101])
       for i in range(len(data)):
           temp_data[i,0:c]=data[i,0]
           temp_data[i,101-d:]=data[i,-1]
       temp_data[:,c:101-d]=data
       return temp_data
###############################################################################
def extend_label(label,a,b,c,d):
    temp_label=np.zeros([len(label)+a+b,len(label[0])+c+d])
    temp_label[a:a+len(label),c:c+len(label[0])]=label
    return temp_label
###############################################################################
def normalize(data):
    data=(data-np.mean(data))/np.std(data)
    return data
###############################################################################
def predict_draw(data,label,m1,m2,threshold1,threshold2,name):
    final_result=[]
    a=max(0,(101-len(data))//2)
    b=max(0,101-len(data)-a)
    c=max(0,(101-len(data[0]))//2)
    d=max(0,101-len(data[0])-c)
    l1=len(data)
    l2=len(data[0])
    data=extend_function(data)
    label=extend_label(label,a,b,c,d)
    cmap = colors.ListedColormap(['white', 'white'])
    bounds = [0,0.5,1]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots()
    ax.imshow(label, cmap=cmap, norm=norm)
    for xo in range(len(label)):
        for yo in range(len(label[0])):
            if label[xo][yo]==1:
                plt.plot(yo+0.5,xo+0.5,'k.', markersize=1)#middle point in a unit grid
                
    ind_1=0
    flag1=0
    while ind_1<=len(data):
        if ind_1==len(data) or flag1==1:
           break
        if ind_1+101>len(data):
            ind_1=len(data)-101
            flag1=1
        ind_2=0
        flag2=0
        while ind_2<=len(data[0]):
            if ind_2==len(data[0]) or flag2==1:
               break
            if ind_2+101>len(data[0]):
                ind_2=len(data[0])-101
                flag2=1
            ###################################################################
            test_data=normalize(data[ind_1:ind_1+101,ind_2:ind_2+101])
            test_data=test_data.reshape(1,101,101,1)
            re=model.predict(test_data)[0]
            loc=np.where(re>threshold1)[0]
            final_loc=[]
            for k in loc:
                y0=ind_1+k//10*10
                x0=ind_2+k%10*10
                rect =patches.Rectangle((x0,y0),10,10,linewidth=0.5, edgecolor='b', facecolor='none')
                ax.add_patch(rect)
                #########################judge if inside the true region or not#########################
                if y0+11<a or y0>a+l1 or x0+11<c or x0>c+l2:#all outside
                    continue
                y0=a if y0<a else y0
                x0=c if x0<c else x0
                y0=a+l1-11 if y0+11>a+l1 else y0
                x0=c+l2-11 if x0+11>c+l2 else x0
                #######################################################################################
                #tmp=test_data[0,k//10*10:k//10*10+11,k%10*10:k%10*10+11,:]
                tmp=data[y0:y0+11,x0:x0+11]
                tmp=(tmp-np.mean(tmp))/np.std(tmp)
                re2=model2.predict(tmp.reshape(1,11,11,1))
                #loc2=np.where(re2[0]>bound2)[0]
                loc2=np.where(re2[0]>threshold2)[0]#debug, to see if times 1000, still can detect
                for j in loc2:
                    y2=j//10
                    x2=j%10
                    final_loc.append([x0+x2,y0+y2])
    
            for z in final_loc:
                rect =patches.Rectangle((z[0],z[1]),1,1,linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                final_result.append((z[1],z[0]))
                ###############################################################
            ind_2+=m2
        ind_1+=m1 
    if a!=0 or c!=0:
       rect =patches.Rectangle((c,a),l2-1,l1-1,linewidth=1, edgecolor='gold', facecolor='none')
       ax.add_patch(rect)
    plt.show()
    plt.title("length of each side="+str(name))
    plt.savefig('/2d/2d_general_prediction/plot/CNN+CNN_'+str(name)+'.svg')
    plt.close()    
    print(final_result)
#########################an example######################################################
a=[51,71,101,151,201]
for h in range(5):
    temp=loadmat('D:/paper_material/2d/2d_general_prediction/data_example/'+str(a[h])+'.mat')
    data=temp['Z']
    label=temp['label1']
    predict_draw(data,label,90,90,0.5,0.3,a[h])

###############################################################################
    
#如果框在外边，把框移动到最近的里面最检测。
