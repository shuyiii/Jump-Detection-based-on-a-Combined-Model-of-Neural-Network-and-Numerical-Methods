
import numpy as np
#make n*n grid
n=10
num=76045
folder='/fs/project/PAS1263/data/circle_1mil/'
label_10_circle=[]
for i in range(1,num+1):
    label=[]
    temp=loadmat(folder+'image'+str(i)+'.mat')
    x=temp['label1']
    for j in range(100//n):
        for k in range(100//n):
            if np.sum(x[j*n:(j+1)*n,k*n:(k+1)*n])>0:
                label.append(1)
            else:
                label.append(0)
    label_10_circle.append(label)
np.save(folder+'label_10_circle.npy')   
############################################################################### 
num=68077
folder='/fs/project/PAS1263/data/line_curve/'
label_10_line=[]
for i in range(1,num+1):
    label=[]
    temp=loadmat(folder+'image'+str(i)+'.mat')
    x=temp['label1']
    for j in range(100//n):
        for k in range(100//n):
            if np.sum(x[j*n:(j+1)*n,k*n:(k+1)*n])>0:
                label.append(1)
            else:
                label.append(0)
    label_10_line.append(label)
np.save(folder+'label_10_line.npy')
###############################################################################   

    
    

