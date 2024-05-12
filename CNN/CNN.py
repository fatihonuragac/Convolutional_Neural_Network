# -*- coding: utf-8 -*-
"""
Created on Sun May 12 13:21:13 2024

@author: fatihonuragac
"""

"""
You can find dataset that used in this notebook at this link https://www.kaggle.com/datasets/moltean/fruits

"""
#%%   Import Libraries

import numpy as np
import pandas as pd 
from keras.models import Sequential
from keras.layers import MaxPooling2D,Conv2D,Activation,Dropout,Flatten,Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
import matplotlib.pyplot as plt
from glob import glob 
import warnings
warnings.filterwarnings('ignore')



#%%  Load and Check Data

train_path=("../input/fruits/fruits-360_dataset/fruits-360/Training" )          
test_path=("../input/fruits/fruits-360_dataset/fruits-360/Test" )                           
img=load_img(train_path+"/Apple Braeburn/0_100.jpg")         
plt.imshow(img)
plt.axis("off")
plt.show()
#%%   Find Number of Classes and Image Size

x=img_to_array(img)
print(x.shape)
className=glob(train_path+'/*')    
numberOfClass=len(className)     
print(numberOfClass)
#%%  Create CNN Model

model=Sequential() #Sequential is skleton for the cnn than we can add layers this skeleton.
model.add(Conv2D(32,(3,3),activation ='relu', input_shape = x.shape))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3),activation ='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3),activation ='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
#Add to ann model.
model.add(Dense(1024,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(numberOfClass,activation="softmax"))

model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])
batch_size=32
#%%  Image Data Generator

train_datagen=ImageDataGenerator(rescale=1./255,
                                zoom_range=0.3,
                                horizontal_flip=True,
                                rotation_range=0.3)
test_datagen=ImageDataGenerator(rescale=1./255)
#%% Flow From Directory

train_generator=train_datagen.flow_from_directory(train_path,
                                                 target_size=x.shape[:2],
                                                 batch_size=batch_size,
                                                 color_mode="rgb",
                                                 class_mode="categorical")
test_generator=test_datagen.flow_from_directory(test_path,
                                                 target_size=x.shape[:2],
                                                 batch_size=batch_size,
                                                 color_mode="rgb",
                                                 class_mode="categorical")
#%% Fit the Model

hist=model.fit(train_generator,
                   steps_per_epoch=3200//batch_size,
                   epochs=100,
                   validation_data=test_generator,
                   validation_steps=1600//batch_size)
#%% Save Results

model.save_weights("Save_weights.weights.h5")
import json
with open("Save_accu&lose.json","w") as s:
    json.dump(hist.history,s)

import codecs
with codecs.open("Save_accu&lose.json","r",encoding="utf-8") as s:
    h=json.loads(s.read())
#%% Visualization of Results

print(h.keys()) 
plt.plot(h["loss"],label="Train Lose")
plt.plot(h["val_loss"],label="Validation Lose")
plt.legend()
plt.show()
plt.figure()
plt.plot(h["accuracy"],label="Train Accuracy")
plt.plot(h["val_accuracy"],label="Validation Accuracy")
plt.legend()
plt.show()

























