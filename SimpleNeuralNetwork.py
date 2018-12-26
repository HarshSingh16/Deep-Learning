#Simple Neural Network using Keras


import keras
from keras.models import Sequential
from keras.layers import *


#Building the Model
model=k.models.Sequential()
model.add(Dense(3,input_dim=2))#Input Layer
model.add(Dense(3))
model.add(Dense(1))#Output Layer
model.compile(loss="mean_squared_error",optimizer="adam")


#Training the model
model.fit(training_data,expected_output)


#Testing Phase
error_rate=model.evaluate(testing_data,expected_output)


#Saving the model to a file
model.save("trained_model.h5")


#Evaluating the model to make predictions
model=keras.models.load_model("trained_model.h5")

#Predictions
Predictions=model.predict(new_Data)
