# -*- coding: utf-8 -*-
"""
@author: wang7
2d general detection
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
        
    temp.sort()
    MMLmf = temp[0]        
    return MMLmf
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
                plt.plot(yo+0.5,xo+0.5,'k.', markersize=1)
                
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
                #image=test_data[0,k//10*10:k//10*10+11,k%10*10:k%10*10+11,:]
                image=normalize(data[y0:y0+11,x0:x0+11]).reshape(11,11,1)#because when calculating threshold, we use nomalization on 101*101, if this reasonable???
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
                reconstruct_MML_matrix = abs(np.array(reconstruct_MML_matrix))>threshold2
                B = (reconstruct_MML_matrix!=0)
                loc_0 = np.where(B>0)
                loc_t = list(zip(loc_0[0],loc_0[1]))
                for q in loc_t:
                    y1 = q[0]
                    x1 = q[1]
                    rect = patches.Rectangle((x0+x1,y0+y1),1,1,linewidth = 1,edgecolor = 'r',facecolor = 'none') 
                    final_result.append((y0+y1,x0+x1))
                    ax.add_patch(rect)
                ###############################################################
            ind_2+=m2
        ind_1+=m1 
    if a!=0 or c!=0:
       rect =patches.Rectangle((c,a),l2,l1,linewidth=1, edgecolor='gold', facecolor='none')
       ax.add_patch(rect)
    plt.show()
    plt.title("length of each side="+str(name))
    plt.savefig('D:/paper_material/2d/2d_general_prediction/plot/'+str(name)+'.svg')
    plt.close()    
    print(final_result)
#########################an example######################################################
a=[51,71,101,151,201]
for h in range(5):
    temp=loadmat('/2d/2d_general_prediction/data_example/'+str(a[h])+'.mat')
    data=temp['Z']
    label=temp['label1']
    predict_draw(data,label,90,90,0.4,0.2,a[h])
  
    

            

            
            

