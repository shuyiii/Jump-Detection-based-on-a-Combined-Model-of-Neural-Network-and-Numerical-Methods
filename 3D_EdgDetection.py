#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:20:32 2019

@author: zhou
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import scipy.io
from scipy.io import loadmat
from scipy.spatial import Delaunay

input_image=loadmat('/Users/zhou/Dropbox/MATLAB/sphere cut/3D_ball sphere/image1.mat')

#data = input_image['f']
#image = (data-np.mean(data))/np.std(data)
image = input_image['f']
X = input_image['X']
Y = input_image['Y']
Z = input_image['Z']

max_m = 4  
threshold = 0.1
label = input_image['label1']


#generate reconstruct matrix

nx = image.shape[0]
ny = image.shape[0] 
nz = image.shape[0]
x = np.linspace(0,1,nx)
y = np.linspace(0,1,ny)
z = np.linspace(0,1,nz)

grid_points = []
for fi in x:
    for fj in y:
        for fk in z:
            grid_points.append((fi,fj,fk))

#choose S_(center)


def get_nearest_N_point(grid_points, center):
    
    res = []
            
    for pt in grid_points:
        distance = get_distance(center, pt)
        res.append((distance,pt))
    res = sorted(res)
    N_point = [x[1] for x in res]
    return N_point


    
def get_distance(pt1,pt2):
    temp = 0
    for i in range(len(pt1)):
        temp += (pt1[i]-pt2[i])**2
    return temp**0.5 


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

def get_qm(height,S_Sort,C):
    temp = np.array(height)[1 :]-np.array(height)[:-1]
    max_index_r = np.argmax(temp)
    sub_C = np.array(C)[0:(max_index_r+1)]
    qm  = sum(sub_C)
    return qm
    

def combination(m):
    com = []
    for n in range(m+1):     
        for i in range(n+1):
            j = n-i
            for k in range(j+1):
                l = j-k
                com.append((i,k,l))
    return com
    



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

def get_ls_b(m,alpha_12):
    b = []
    for (a1,a2,a3) in alpha_12:
        if a1+a2+a3 == m:
            val = math.factorial(a1)*math.factorial(a2)*math.factorial(a3)
            b.append(val)
        else:
            b.append(0)
    return b


    
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



reconstruct_MML_matrix = np.zeros((nx-1,ny-1,nz-1))
for k in range(nz-1):
    for i in range(nx-1):
        for j in range(ny-1):
            
            center_six = cube_points_center(i,j,k,x,y,z)
            mml_center =[]
            for center in center_six:
                mml_center.append(MMLmf(max_m,x,y,z,image,center,grid_points))
                
            reconstruct_MML_matrix[i][j][k] = max(mml_center)
    



MML_matrix = reconstruct_MML_matrix>threshold
#
B = (MML_matrix!=0)
loc = np.where(B>0)
loc_t = list(zip(loc[0],loc[1],loc[2]))
#
           
#            
###TPR
#
n = len(label[0])
total = 0
for i in range(n):
    for j in range(n):
        for k in range(n):
            if label[i][j][k]==1:
                total+=1

TP = 0
for k in loc_t:
    if label[k[0]][k[1]][k[2]]!=0:
        TP+=1
TPR = TP/total

FP = 0
for k in loc_t:
    if label[k[0]][k[1]][k[2]]!=1:
        FP+=1
FPR = FP/(n*n*n-total)
##
##ROC curve  
PR_point = 100
threshold = np.linspace(0,0.5,nx)
TPRY =[]
FPRX = []
for th in threshold:
    MML_matrix = reconstruct_MML_matrix>th
    B = (MML_matrix!=0)
    loc = np.where(B>0)
    loc_t = list(zip(loc[0],loc[1],loc[2]))
    n = len(label[0])
    total = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if label[i][j][k]==1:
                    total+=1
    
    TP = 0
    for k in loc_t:
        if label[k[0]][k[1]][k[2]]!=0:
            TP+=1
    TPR = TP/total
    
    FP = 0
    for k in loc_t:
        if label[k[0]][k[1]][k[2]]!=1:
            FP+=1
    FPR = FP/(n*n*n-total)
    
    TPRY.append(TPR)
    FPRX.append(FPR)
    
plt.plot(FPRX,TPRY)
plt.xlabel('FPR')
plt.ylabel('TPR')