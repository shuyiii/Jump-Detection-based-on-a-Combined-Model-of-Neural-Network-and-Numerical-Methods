# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 19:09:48 2019

@author: wang7
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import keras
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Sequential
from keras.layers import Conv3D, MaxPool3D,Lambda,Dense,BatchNormalization,Flatten,Reshape,Dropout,Activation
from keras.optimizers import SGD, Adam
from scipy.io import loadmat
from matplotlib import *
import sys
from keras.models import Model
import tensorflow as tf
from matplotlib import colors
import matplotlib.patches as patches
import math
from scipy.spatial import Delaunay
###############################################################################
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
#for one center point
def sort_near_center(S_center, x, y, z , image):
    a = []
    for (i1,i2,i3) in S_center:
        i = list(x).index(i1)
        j = list(y).index(i2)
        k = list(z).index(i3)
        f = image[i][j][k]
        a.append((f,(i1,i2,i3)))
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
            for k in range(j+1):
                l = j-k
                com.append((i,k,l))
    return com
###############################################################################
def get_ls_matrix(S_center,alpha_12):
    ls_matrix = []
    for (a1,a2,a3) in alpha_12:
        row = []
        for (u,v,w) in S_center:
            val = (u**a1)*(v**a2)*(w**a3)
            row.append(val)
        ls_matrix.append(np.array(row))
#    print(np.linalg.det(np.array(ls_matrix)))
    return ls_matrix
#np.linalg.det(np.array(ls_matrix))
###############################################################################
def get_ls_b(m,alpha_12):
    b = []
    for (a1,a2,a3) in alpha_12:
        if a1+a2+a3 == m:
            val = math.factorial(a1)*math.factorial(a2)*math.factorial(a3)
            b.append(val)
        else:
            b.append(0)
    return b
###############################################################################
def MMLmf(max_m,x,y,z,image,center,grid_points):
    near_center = get_nearest_N_point(grid_points, center)
    temp = []
    for m in range(1,max_m+1):
        n = (m+3)*(m+2)*(m+1)/(2*3)
        S_center = near_center[:int(n)]
        height_sortband_location = sort_near_center(S_center, x, y, z,image)
        S_Sort = [x[1] for x in height_sortband_location]
        height = [x[0] for x in height_sortband_location]
    #Ax = b  (PC = 0!)
        alpha_123 = combination(m)
        ls_matrix_A = get_ls_matrix(S_Sort,alpha_123)
        ls_b = get_ls_b(m,alpha_123)
        C = np.linalg.lstsq(ls_matrix_A, ls_b,rcond=None)
        C = C[0]
            #get qm
        qm = get_qm(height,S_Sort,C)
        Lmf_center = abs(float(1)/qm*np.dot(np.array(C),height))
        temp.append(Lmf_center)
    temp.sort()
#    print(temp)
    MMLmf = temp[0]   
    return MMLmf       
###############################################################################
def cube_points_center(i,j,k,x,y,z):
    points =[]
    for a in (x[i],x[i+1]):
        for b in (y[j],y[j+1]):
            for c in (z[k],z[k+1]):
                points.append([a,b,c])
    tri = Delaunay(points)
    center =[]
    for vertex in tri.simplices:
        vertex_location = [points[i] for i in vertex]
        center.append(np.mean(vertex_location, axis=0))
    return center
###############################################################################
folder='D:/sphere_cut/'
test_data=[]
test_label=[]
for i in range(5000,5021):    
    temp=loadmat(folder+'image'+str(i)+'.mat')
    tmp=temp['f']
    tmp=(tmp-np.mean(tmp))/np.std(tmp)
    test_data.append(tmp.reshape(101,101,101,1))#K.image_dim_ordering()='tf'
    test_label.append(temp['label1'])

test_data=np.asarray(test_data)
test_label=np.asarray(test_label)
###############################################################################
model = Sequential()
model.add(Conv3D(12, (5,5,5),strides=(5,5,5),input_shape=(101,101,101,1),activation='relu'))
model.add(MaxPool3D(pool_size=(2,2,2)))
model.add(Conv3D(6, (2,2,2),strides=(2,2,2),activation='relu'))
model.add(Flatten())
model.add(Dense(1000))
model.load_weights('D:/paper_material/3d/model_weights_3D_10_10.h5')
result=model.predict(test_data)
###############################################################################
threshold1=0.4
threshold2=0.1
for i in range(21):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    true_loc=np.where(test_label[i]==1)#should be array
    #ax.scatter3D(true_loc[2]+0.5, true_loc[1]+0.5, true_loc[0]+0.5, c=true_loc[0])
    x0,y0,z0=[],[],[]
    pred_loc=np.where(result[i]>threshold1)[0]
    for loc in pred_loc:
        z0.append(loc//100*10)
        x0.append(loc%100%10*10)
        y0.append(loc%100//10*10)
    for j in range(len(x0)):
        image=test_data[i][z0[j]:z0[j]+11,y0[j]:y0[j]+11,x0[j]:x0[j]+11]
#generate reconstruct matrix
        nx = image.shape[2]
        ny = image.shape[1] 
        nz = image.shape[0]
        x = np.linspace(0,1,nx)
        y = np.linspace(0,1,ny)
        z = np.linspace(0,1,nz)
        grid_points = []
        for fi in x:
            for fj in y:
                for fk in z:
                    grid_points.append((fi,fj,fk))
        reconstruct_MML_matrix = np.zeros((nx-1,ny-1,nz-1))
        for k in range(nz-1):
            for m in range(nx-1):
                for l in range(ny-1):
                    center_six = cube_points_center(m,l,k,x,y,z)
                    mml_center =[]
                    for center in center_six:
                        mml_center.append(MMLmf(3,x,y,z,image,center,grid_points))     
                    reconstruct_MML_matrix[m][l][k] = max(mml_center)
        MML_matrix = reconstruct_MML_matrix>threshold2
        B = (MML_matrix!=0)
        loc = np.where(B>0)
        loc_t = list(zip(loc[0],loc[1],loc[2]))
        for q in loc_t:
            z1 = q[0]
            y1 = q[1]
            x1 = q[2]
            ax.scatter3D(x0[j]+x1+0.5, y0[j]+y1+0.5, z0[j]+z1+0.5)
    plt.show()
    plt.savefig('D:/paper_material/3d/plot/'+str(i)+'_2step'+'.svg')
    plt.close()