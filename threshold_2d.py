# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 01:25:53 2019

@author: wang7
"""
from numpy import linalg as LA
import tensorflow as tf
import keras
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Sequential
from keras.layers import Lambda,Dense,BatchNormalization,Flatten,Reshape,Dropout, Activation,Convolution2D,GlobalAveragePooling1D
from keras.layers import MaxPooling2D,Convolution2D
from keras.optimizers import SGD, Adam
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import *
import sys
import pylab as pl
from keras.models import Model
import math
###############################################################################
"2D threshold chosen for 1 step method"
#######function definition#####################################################
def get_nearest_N_point(grid_points, center):
    
    res = []
            
    for pt in grid_points:
        distance = get_distance(center, pt)
        res.append((distance,pt))
    res = sorted(res)
    N_point = [x[1] for x in res]
    return N_point
############################################################################### 
def get_distance(pt1,pt2):
    temp = 0
    for i in range(len(pt1)):
        temp += (pt1[i]-pt2[i])**2
    return temp**0.5 
###############################################################################
def sort_near_center(S_center, x, y, image):
    a = []
    for (i1,i2) in S_center:
        i = list(x).index(i1)
        j = list(y).index(i2)
        f = image[j][i]
        a.append((f,(i1,i2)))
    A = sorted(a)
    return A
###############################################################################
def get_qm(height,S_Sort,C):
    temp = np.array(height)[1 :]-np.array(height)[:-1]
    max_index_r = np.argmax(temp)
    sub_C = np.array(C)[0:(max_index_r+1)]
    qm  = sum(sub_C)
    return qm
###############################################################################
def combination(m):
    com = []
    for n in range(m+1):
        
        for i in range(n+1):
            j = n-i
            com.append((i,j))
    return com
###############################################################################
def get_ls_matrix(S_center,alpha_12):
    ls_matrix = []
    for (a1,a2) in alpha_12:
        row = []
        for (u,v) in S_center:
            val = (u**a1)*(v**a2)
            row.append(val)
        ls_matrix.append(row)
    return ls_matrix
###############################################################################
def get_ls_b(m,alpha_12):
    b = []
    for (a1,a2) in alpha_12:
        if a1+a2 == m:
            val = math.factorial(a1)*math.factorial(a2)
            b.append(val)
        else:
            b.append(0)
    return b
############################################################################### 
def MMLmf(max_m,x,y,image,center_right,center_left):

    near_center_right = get_nearest_N_point(grid_points, center_right)
    near_center_left = get_nearest_N_point(grid_points, center_left)
    temp = []
    for m in range(1,max_m+1):
    
        n = (m+2)*(m+1)/2
        
        S_center_right = near_center_right[:int(n)]
        S_center_left = near_center_left[:int(n)]
        height_sortband_location_right = sort_near_center(S_center_right, x, y, image)
        height_sortband_location_left = sort_near_center(S_center_left, x, y, image)
        S_Sort_right = [x[1] for x in height_sortband_location_right]
        height_right = [x[0] for x in height_sortband_location_right]
        S_Sort_left = [x[1] for x in height_sortband_location_left]
        height_left = [x[0] for x in height_sortband_location_left]
    #Ax = b  (PC = 0!)
        alpha_12 = combination(m)
        ls_matrix_A_right = get_ls_matrix(S_Sort_right,alpha_12)
        ls_matrix_A_left = get_ls_matrix(S_Sort_left,alpha_12)
        ls_b = get_ls_b(m,alpha_12)
        C_right = np.linalg.lstsq(ls_matrix_A_right, ls_b,rcond=None)
        C_right = C_right[0]
        C_left = np.linalg.lstsq(ls_matrix_A_left, ls_b,rcond=None)
        C_left = C_left[0]
            #get qm
        qm_right = get_qm(height_right,S_Sort_right,C_right)
        qm_left = get_qm(height_left,S_Sort_left,C_left)
        Lmf_center_right = abs(float(1)/qm_right*np.dot(np.array(C_right),height_right))
        Lmf_center_left = abs(float(1)/qm_left*np.dot(np.array(C_left),height_left))
        temp.append(max(Lmf_center_right+Lmf_center_left))
        
#    print('mml =', temp)
#    print('qm_right = ', qm_right)
#    print('qm_left = ', qm_left)
    temp.sort()
    MMLmf = temp[0]        
    return MMLmf

#######load data########################################################################
folder1='/users/PAS1263/osu8085/2D_data/circle_1wan/circle_1wan/'
folder2='/users/PAS1263/osu8085/2D_data/line_test/line_test/'
folder='/users/PAS1263/osu8085/2D_models/new_train/model_weights_10_10_normalization_each.h5'#load model

test_data=[]
test_label=[]
test_label_100=[]
n=10
for i in range(1,10001):
    label=[]
    temp=loadmat(folder1+'image'+str(i)+'.mat')
    img=temp['image']
    img=(img-np.mean(img))/np.std(img)
    test_data.append(img.reshape(101,101,1))
    x=temp['label1']
    test_label_100.append(temp['label1'])
    for j in range(100//n):
        for k in range(100//n):
            if np.sum(x[j*n:(j+1)*n,k*n:(k+1)*n])>0:
                label.append(1)
            else:
                label.append(0)
    test_label.append(label)
    label=[]
    temp=loadmat(folder2+'image'+str(i)+'.mat')
    img=temp['image']
    img=(img-np.mean(img))/np.std(img)
    test_data.append(img.reshape(101,101,1))
    x=temp['label1']
    test_label_100.append(temp['label1'])
    for j in range(100//n):
        for k in range(100//n):
            if np.sum(x[j*n:(j+1)*n,k*n:(k+1)*n])>0:
                label.append(1)
            else:
                label.append(0)
    test_label.append(label)#because no label with 10*10, when give code, also provide generating label, since when choose, may different labels size

test_data=np.asarray(test_data)
test_label=np.asarray(test_label)
######load model#########################################################################
model = Sequential()
model.add(Convolution2D(32, (4,4),strides=(1,1),input_shape=(101,101,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
model.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
model.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.load_weights(folder)
###############################################################################
result=model.predict(test_data)
theshold=0
total=np.sum(test_label==1)
minimum=1
for i in range(1,10001):
    thresh=i*0.0001 
    num=np.sum(result>thresh)
    if abs(num/total-1)<minimum:
        minimum=abs(num/total-1)
        threshold=thresh
print(threshold)
print(minimum)
#threshold=0.3668
#minimum=0.0
###############################################################################
#after 1st stage, total jumps in detected large grids
v=result>threshold
total=0
for i in range(20000):
    for j in range(100):
        if v[i][j]:
           total+=np.sum(test_label_100[i][j//n*10:(j//n+1)*10,j%n*10:(j%n+1)*10]==1)
###############################################################################
# numerical threshold after 1st stage threshold decided
###############################################################################
original_imsize =100
nx = 11#image.shape[1]
ny = 11#image.shape[0] 

x = np.linspace(0,(nx-1)/original_imsize,nx)
y = np.linspace((ny-1)/original_imsize,0,ny)
u = 0.4*x[0:-1]+0.6*x[1:]
v = 0.4*y[0:-1] +0.6*y[1:]
r = 0.6*x[0:-1]+0.4*x[1:]
s = 0.6*y[0:-1] +0.4*y[1:]
xx,yy = np.meshgrid(x,y)
uu,vv = np.meshgrid(u,v)
rr,ss = np.meshgrid(r,s)
grid_points = list(zip(np.append([],xx),np.append([],yy)))
center_points_right = list(zip(np.append([],uu),np.append([],vv)))
center_points_left = list(zip(np.append([],rr),np.append([],ss)))
######plot loop#########################################################################
threshold1=threshold
threshold2=0
minimum=1
final_re=[]
for n1 in range(20000):
        data=test_label_100[n1]
        re=result[n1]
        loc=np.where(re>threshold1)[0]
        for k in loc:
            final_re.append((n1,k))

for i in range(1,101):
    thresh=i*0.01 
    num=0
    for d in final_re:
            image=test_data[d[0]][d[1]//10*10:d[1]//10*10+11,d[1]%10*10:d[1]%10*10+11]
            max_m = 3
            temp = 0
            reconstruct_MML_matrix = []
            for i in range(nx-1):
                row = []
                for j in range(ny-1):
                    center_right = center_points_right[temp]
                    center_left = center_points_left[temp]
                    temp += 1
                    mml_center = MMLmf(max_m,x,y,image,center_right,center_left)
                    row.append(mml_center)
                reconstruct_MML_matrix.append(row)
        
            reconstruct_MML_matrix = abs(np.array(reconstruct_MML_matrix))>thresh
            B = (reconstruct_MML_matrix!=0)
            loc_0 = np.where(B>0)
            num+=len(loc_0[0])
    if abs(num/total-1)<minimum:
        minimum=abs(num/total-1)
        threshold2=thresh
print(threshold2)
print(minimum)
###############################################################################
#10-10 CNN threshold2
model2 = Sequential()
model2.add(Convolution2D(32, (2,2),strides=(1,1),input_shape=(11,11,1),activation='relu'))
model2.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
model2.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
model2.add(Flatten())
model2.add(Dense(100))
model2.load_weights('/users/PAS1263/osu8085/2D_models/new_train/model_weights_10_10_local_norm.h5')
###############################################################################
result=model.predict(test_data)
final_re=[]
threshold1=threshold
threshold2=0
minimum=1
for i in range(20000):
    re=result[i] 
    loc=np.where(re>threshold1)[0]
    for k in loc:
        y=k//10*10
        x=k%10*10
        re2=model2.predict(test_data[i][y:y+11,x:x+11].reshape(1,11,11,1))
        final_re.append(re2[0])
        
for i in range(1,101):
    thresh=i*0.01 
    num=0
    for j in range(len(final_re)):
        loc2=np.where(final_re[j]>thresh)[0]
        num+=len(loc2)
    if abs(num/total-1)<minimum:
        minimum=abs(num/total-1)
        threshold2=thresh
print(threshold2)
print(minimum)
###############################################################################
#later cross validation

