#Simple Neural Network using Keras

from keras.models import Sequential
from keras.layers import *

model=k.models.Sequential()
model.add(Dense(3,input_dim=2))#Input Layer
model.add(Dense(3))
model.add(Dense(1))#Output Layer
model.compile(loss="mean_squared_error",optimizer="adam")
