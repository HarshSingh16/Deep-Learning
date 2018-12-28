#Import the necessay packages

import numpy as np
from keras.preprocessing import image
from keras.applications import vgg16
import matplotlib.pyplot as plt
import cv2
import os, os.path

#Load the vgg model from keras
model = vgg16.VGG16()

 
#image path and valid extensions- Setting the directory where we have stored the images
imageDir = "/Users/harshsingh/Downloads/Ex_Files_Deep_Learning_Image_Recog/Exercise Files/Ch04/untitled folder" #specify your path here
image_path_list = []
valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"] #specify your vald extensions here
valid_image_extensions = [item.lower() for item in valid_image_extensions]
 
#create a list all files in directory and append files with a vaild extention to image_path_list
for file in os.listdir(imageDir):
    extension = os.path.splitext(file)[1]
    if extension.lower() not in valid_image_extensions:
        continue
    image_path_list.append(os.path.join(imageDir, file))
 
#loop through image_path_list to open each image and subsequently throw in the predictions
for imagePath in image_path_list:
    image = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
    width = 224
    height = 224
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    list_of_images = np.expand_dims(resized, axis=0)
    x = vgg16.preprocess_input(list_of_images)
    predictions = model.predict(x)
    predicted_classes = vgg16.decode_predictions(predictions)
    print("Top predictions for this image:")
    for imagenet_id, name, likelihood in predicted_classes[0]:
        print("Prediction: {} - {:2f}".format(name, likelihood))
    plt.imshow(image)
    # Show the plot on the screen
    plt.show()
