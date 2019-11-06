# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 01:27:37 2019

@author: wang7
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from keras import backend as K
from keras.models import Sequential
from keras.layers import MaxPooling2D,Lambda,Dense,BatchNormalization,Flatten,Reshape,Dropout,Activation,Convolution2D,GlobalAveragePooling1D
from keras.optimizers import SGD, Adam
from scipy.io import loadmat
from keras.models import Model
from sklearn import preprocessing
from matplotlib import colors
from scipy.spatial import Delaunay
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
def cube_points_center(i,j,x,y):
    points =[]
    for a in (x[i],x[i+1]):
        for b in (y[j],y[j+1]):      
            points.append([a,b])   
    tri = Delaunay(points)
    center =[]
    for vertex in tri.simplices:
        vertex_location = [points[i] for i in vertex]
        center.append(np.mean(vertex_location, axis=0))
    return center
###############################################################################
def MMLmf(max_m,x,y,image,center,grid_points):
    near_center = get_nearest_N_point(grid_points, center)
    temp = []
    for m in range(1,max_m+1):
        n = (m+2)*(m+1)/2
        S_center = near_center[:int(n)]
        height_sortband_location = sort_near_center(S_center, x, y,image)
        S_Sort = [x[1] for x in height_sortband_location]
        height = [x[0] for x in height_sortband_location]
        alpha_123 = combination(m)
        ls_matrix_A = get_ls_matrix(S_Sort,alpha_123)   
        ls_b = get_ls_b(m,alpha_123)
        C = np.linalg.lstsq(ls_matrix_A, ls_b,rcond=None)
        C = C[0]
        qm = get_qm(height,S_Sort,C)      
        Lmf_center = float(1)/qm*np.dot(np.array(C),height)
        temp.append(Lmf_center)

    temp.sort()
    if temp[0]>0:
        MMLmf  = temp[0]
    elif temp[-1]<0:
        MMLmf = abs(temp[-1])
    else:
        MMLmf=0    
    return MMLmf
#######load data########################################################################
folder1='D:/paper_material/2d/circle curve/circle curve/circle/'
folder2='D:/paper_material/2d/one line curve/one line curve/one line/'
folder='E:/CNN/detect discontinuity/2D data/new_models/model_weights_10_10_normalization_each.h5'#load model

test_data=[]
test_label=[]
for i in range(1,20):
    temp=loadmat(folder1+'image'+str(i)+'.mat')
    img=temp['image']
    img=(img-np.mean(img))/np.std(img)
    test_data.append(img.reshape(101,101,1))
    test_label.append(temp['label1'].ravel())
    temp=loadmat(folder2+'image'+str(i)+'.mat')
    img=temp['image']
    img=(img-np.mean(img))/np.std(img)
    test_data.append(img.reshape(101,101,1))
    test_label.append(temp['label1'].ravel())

test_data=np.asarray(test_data)
test_label=np.asarray(test_label)
######load model#########################################################################
model = Sequential()
model.add(Convolution2D(32, (4,4),strides=(1,1),input_shape=(101,101,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
model.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
model.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100))
model.load_weights(folder)
###############################################################################
result=model.predict(test_data)
threshold=0.35
######plot loop#########################################################################
for n in range(20):
    wrong_data=[]
    wrong_label=[]
    data=test_label[n].reshape(100,100)
    re=result[n]
    loc=np.where(re>threshold)[0] 
    cmap = colors.ListedColormap(['white', 'white'])
    bounds = [0,0.5,1]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots()
    ax.imshow(data, cmap=cmap, norm=norm)
    for xo in range(100):
        for yo in range(100):
            if data[xo][yo]==1:
                plt.plot(yo,xo,'k.', markersize=1)
    for k in loc:
        y0=-0.5+k//10*10
        x0=-0.5+k%10*10
        rect =patches.Rectangle((x0,y0),10,10,linewidth=0.5, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        image=test_data[n][k//10*10:k//10*10+11,k%10*10:k%10*10+11]
        ##numerical coef
        max_m = 3
        threshold2 = 0.1
        original_imsize =100#should it be original_imsize =(image.shape[1]-1)*(image.shape[0]-1)???
        nx = image.shape[1]
        nx = image.shape[1]
        ny = image.shape[0] 
        x = np.linspace(0,(nx-1)/original_imsize,nx)
        y = np.linspace((ny-1)/original_imsize,0,ny)
        xx,yy = np.meshgrid(x,y)
        grid_points = list(zip(np.append([],xx),np.append([],yy)))    
        temp = 0
        reconstruct_MML_matrix = np.zeros((ny-1,nx-1))
        for i in range(nx-1):
            for j in range(ny-1):
                center_two = cube_points_center(i,j,x,y)
                mml_center =[]
                for center in center_two:
                    mml_center.append(MMLmf(max_m,x,y,image,center,grid_points))                  
                reconstruct_MML_matrix[j][i] = max(mml_center) 
        reconstruct_MML_matrix = abs(np.array(reconstruct_MML_matrix))>threshold2#if need abs or not??
        B = (reconstruct_MML_matrix!=0)
        loc_0 = np.where(B>0)
        loc_t = list(zip(loc_0[0],loc_0[1]))
        for q in loc_t:
            y1 = q[0]
            x1 = q[1]
            rect = patches.Rectangle((x0+x1,y0+y1),1,1,linewidth = 1,edgecolor = 'r',facecolor = 'none') 
            ax.add_patch(rect)
    plt.show()
    plt.savefig('D:/paper_material/2d/2d_general_prediction/plot/'+str(n)+'combine_local'+'.svg')
    plt.close()
