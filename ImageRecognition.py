#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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


# Load data set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize data set to 0-to-1 range
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

#Now we have to make sure that the training and test labels are 2 categorical values- i.e 1 is label is true and 0 otherwise
y_train=keras.utils.to_categorical(y_train,10)
y_test=keras.utils.to_categorical(y_test,10)

# Create a model and add layers
model = Sequential()

model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(512, activation="relu"))
model.add(Dropout(0.50))
model.add(Dense(10, activation="softmax"))


# Print a summary of the model
model.summary()



model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))


# Print a summary of the model
model.summary()




