from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPool3D,Lambda,Dense,BatchNormalization,Flatten,Reshape,Dropout,Activation
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
from scipy.io import loadmat
from keras.callbacks import ModelCheckpoint

num=1962
folder='/users/PAS1263/osu8085/sphere_cut/sphere_cut/3D_ball sphere/'
train_data=[]
train_label=[]
for i in range(1,num+1):
    temp=loadmat(folder+'image'+str(i)+'.mat')
    tmp=temp['f']
    tmp=(tmp-np.mean(tmp))/np.std(tmp)
    train_data.append(tmp.reshape(101,101,101,1))#K.image_dim_ordering()='tf'
    train_label.append(temp['label10'].ravel())

train_data=np.asarray(train_data)
train_label=np.asarray(train_label)
###############################################################################################
def root_mean_squared_error(y_pred,y_true):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

model = Sequential()
model.add(Conv3D(12, (5,5,5),strides=(5,5,5),input_shape=(101,101,101,1),activation='relu'))
model.add(MaxPool3D(pool_size=(2,2,2)))
model.add(Conv3D(6, (2,2,2),strides=(2,2,2),activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1000))

adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer = "adam", loss = root_mean_squared_error, 
              metrics =["accuracy"])
filepath="/users/PAS1263/osu8085/3D_detection/"
filename=filepath+'model_weights_3D_10_10.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=0, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit(train_data,train_label, validation_split=0.1，callbacks=callbacks_list,batch_size=200, epochs=100, verbose=0)
