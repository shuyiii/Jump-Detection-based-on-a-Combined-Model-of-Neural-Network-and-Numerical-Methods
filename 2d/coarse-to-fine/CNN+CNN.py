#folder1='E:/CNN/detect discontinuity/2D data/2D data/circle_1wan/circle_1wan/'
#folder2='E:/CNN/detect discontinuity/2D data/2D data/line_test/line_test/'
folder1='D:/paper_material/2d/test data/circle/'
folder2='D:/paper_material/2d/test data/line/'
folder='E:/CNN/detect discontinuity/2D data/new_models/'#load model

test_data=[]
test_label=[]
for i in range(1,51):
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

model2 = Sequential()
model2.add(Convolution2D(32, (2,2),strides=(1,1),input_shape=(11,11,1),activation='relu'))
model2.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
model2.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
model2.add(Flatten())
model2.add(Dense(100))

model2.load_weights('D:/paper_material/new_2d_functions/model_weights_10_10_local_normalize_3.0.hdf5')
###################################################################################
#first model to 25*25,each 4*4
model1 = Sequential()
model1.add(Convolution2D(32, (4,4),strides=(1,1),input_shape=(101,101,1),activation='relu'))
model1.add(MaxPooling2D(pool_size=(2,2)))
model1.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
model1.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
model1.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
model1.add(Flatten())
model1.add(Dense(100))

model1.load_weights('D:/paper_material/new_2d_functions/model_weights_10_10_normalization_each.h5')
###################################################################################
#2d plot
result=model1.predict(test_data)
bound=0.28#0.4459
bound2=0.22#0.31
final_loc=[]
for i in range(len(test_data)):
    temp=[]
    re=result[i] 
    loc=np.where(re>bound)[0]
    for k in loc:
        y=k//10*10
        x=k%10*10
        tmp=test_data[i][y:y+11,x:x+11]
        tmp=(tmp-np.mean(tmp))/np.std(tmp)
        re2=model2.predict(tmp.reshape(1,11,11,1))
        #loc2=np.where(re2[0]>bound2)[0]
        loc2=np.where(re2[0]>bound2)[0]#debug, to see if times 1000, still can detect
        for j in loc2:
            y2=j//10
            x2=j%10
            temp.append([x+x2,y+y2])
    final_loc.append(temp)


###############################################################################
#folder1='E:/CNN/detect discontinuity/2D data/2D data/circle_1wan/circle_1wan/'
#folder2='E:/CNN/detect discontinuity/2D data/2D data/line_test/line_test/'
folder1='D:/paper_material/2d/test data/circle/'
folder2='D:/paper_material/2d/test data/line/'

test_label_100=[]
for i in range(1,51):
    temp=loadmat(folder1+'image'+str(i)+'.mat')
    test_label_100.append(temp['label1'].ravel())
    temp=loadmat(folder2+'image'+str(i)+'.mat')
    test_label_100.append(temp['label1'].ravel())
test_label_100=np.asarray(test_label_100)
###############################################################################
for i in range(10):
###############################################################################
    data=test_label_100[i].reshape(100,100)
    re=result[i]
    loc=np.where(re>bound)[0]
# create discrete colormap
    cmap = colors.ListedColormap(['white', 'white'])
    bounds = [0,0.5,1]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots()
    fig.set_size_inches(8,8) 
    ax.imshow(data, cmap=cmap, norm=norm,extent=[0,100,100,0])
    for xo in range(100):
        for yo in range(100):
            if data[xo][yo]==1:
                plt.plot(yo+0.5,xo+0.5,'k.', markersize=1)
# draw gridlines
    #ax.grid(which='major', axis='both', linestyle='-', color='y', linewidth=0.5)
    #ax.set_xticks(np.arange(-0.5, 100, 1));
    #ax.set_yticks(np.arange(-0.5, 100, 1));
#draw bounding boxes of predictions
    for k in loc:
        y0=k//10*10#from -0.5 default
        x0=k%10*10
        rect =patches.Rectangle((x0,y0),10,10,linewidth=0.5, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
    for z in final_loc[i]:
        rect =patches.Rectangle((z[0],z[1]),1,1,linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.subplots_adjust(top=0.95,bottom=0.05,left=0.05,right=0.95,hspace=0,wspace=0) 
    plt.margins(0.01,0.01)
    plt.show()
    plt.savefig('D:/paper_material/paper/figure/2CNNs ('+str(i)+').pdf')
    plt.savefig('D:/paper_material/paper/original figures/2CNNs ('+str(i)+').svg')
    plt.close()
###############################################################################

    