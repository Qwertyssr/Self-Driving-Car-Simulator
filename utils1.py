import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
import random

def getName(filePath):
    return filePath.split('/')[-1];
def importDataInfo(path):
    columns=['Center','Left','Right','Steering','Throttle','Brake','Speed']
    #print(os.path.join(path,'driving_log.csv'))
    data=pd.read_csv(os.path.join(path,'driving_log.csv'),names = columns)
    #print(data.head())
    #print(data['Center'][0])
    data['Center']=data['Center'].apply(getName);
    #print(data.head())
    print("Total images imported:",data.shape[0])
    return data
    
def balanceData(data,display=True):
    nBins=31;
    samplesPerBin=1000;
    hist,bins=np.histogram(data['Steering'],nBins);
    center= (bins[:-1]+bins[1:])*0.5;
    removedIndexList=[]
    for i in range(nBins):
        binDataList=[]
        for j in range(len(data['Steering'])):
            if data['Steering'][j]>=bins[i] and data['Steering'][j]<=bins[i+1]:
                binDataList.append(j)
        binDataList=shuffle(binDataList)
        binDataList=binDataList[samplesPerBin:]
        removedIndexList.extend(binDataList)
    data.drop(data.index[removedIndexList],inplace=True)
    print("Remaining images:",len(data))
    if display:
        hist,_=np.histogram(data['Steering'],nBins);
        plt.bar(center,hist,width=0.06);
        plt.plot([-1,1],[samplesPerBin,samplesPerBin]);
        plt.show();
    return data;
    
def loadData(path,data):
    imagesPath=[]
    steering=[]
    for i in range(len(data)):
        indexedData=data.iloc[i];
        #print(indexedData)
        imagesPath.append(os.path.join(path,'IMG',indexedData.iloc[0]))
        steering.append(float(indexedData.iloc[3]))
        #print(os.path.join(path,'IMG',indexedData.iloc[0]));
    imagesPath=np.asarray(imagesPath);
    steering=np.asarray(steering);
    return imagesPath,steering;

def augmentImage(imgPath,steering):
    img=mpimg.imread(imgPath)
    if np.random.rand()<0.5:
        pan=iaa.Affine(translate_percent={'x':(-0.1,0.1),'y':(-0.1,0.1)})
        img=pan.augment_image(img)
    #ZOOM
    if np.random.rand()<0.5:
        zoom=iaa.Affine(scale=(1,1.2))
        img=zoom.augment_image(img)
    #brightness
    if np.random.rand()<0.5:
        brightness=iaa.Multiply((0.4,1.2))
        img=brightness.augment_image(img)
    #flip
    if np.random.rand()<0.5:
        img=cv2.flip(img,1)
        steering= -steering
    return img,steering
    
def preProcessing(img):
    img=img[60:135,:,:]
    img=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img=cv2.GaussianBlur(img,(3,3),0)
    img=cv2.resize(img,(200,66))
    img=img/255
    print(img.shape)
    return img

def batchGen(imagesPath,steeringList,batchSize,trainFlag):
    while True:
        imgBatch=[]
        steeringBatch=[]
        for i in range(batchSize):
            index=random.randint(0,len(imagesPath)-1);
            if trainFlag:
                img,steering=augmentImage(imagesPath[index],steering[index])
            else:
                img=mpimg.imread(imagesPath[index])
                steering=steeringList[index]
            img=preProcessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield(np.asarray(imgBatch),np.asarray(steeringBatch))
            
def creatModel():
    model= Sequential();
    model.add(Convolution2D(24,(5,5),(2,2),input_shape=(66,200,3),activation='elu'))
    model.add(Convolution2D(36,(5,5),(2,2),activation='elu'))
    model.add(Convolution2D(48,(5,5),(2,2),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))
    model.add(Flatten())
    model.add(Dense(100,activations='elu'))
    model.add(Dense(50,activations='elu'))
    model.add(Dense(10,activations='elu'))
    model.add(Dense(1,activations='elu'))
    model.compile(Adam(lr=0.0001),loss='mse')
    return model


