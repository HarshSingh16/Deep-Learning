"""
Created on Wed Dec 26 05:21:28 2018

@author: harshsingh
"""

#Importing the necessary libraries
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import *
from pathlib import Path


#Load the cifar Dataset
(x_train,y_train),(x_test,y_test)=cifar10.load_data()


#Normalize the data- change values from integers to float and make sure that they are between 0 and 1
x_train=x_train.astype("float32")
x_test=y_train.astype("float32")
x_train=x_train/255
x_test=x_test/255

#Now we have to make sure that the training and test labels are 2 categorical values- i.e 1 is label is true and 0 otherwise
x_train=keras.utils.to_categorical(x_train,10)
y_train=keras.utils.to_categorical(y_train,10)
