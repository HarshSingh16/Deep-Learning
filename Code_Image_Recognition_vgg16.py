# Load the json file that contains the model's structure
#f = Path("model_structure.json")
#model_structure = f.read_text()
import numpy as np
from keras.preprocessing import image
from keras.applications import vgg16
# Recreate the Keras model object from the json data
#model = model_from_json(model_structure)
model = vgg16.VGG16()
# Re-load the model's trained weights
#model.load_weights("model_weights1.h5")


# import the necessary packages
import cv2
import os, os.path
 
#debug info OpenCV version
print ("OpenCV version: " + cv2.__version__)
 
#image path and valid extensions
imageDir = "/Users/harshsingh/Downloads/Ex_Files_Deep_Learning_Image_Recog/Exercise Files/Ch04/untitled folder" #specify your path here
image_path_list = []
valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"] #specify your vald extensions here
valid_image_extensions = [item.lower() for item in valid_image_extensions]
 
#create a list all files in directory and
#append files with a vaild extention to image_path_list
for file in os.listdir(imageDir):
    extension = os.path.splitext(file)[1]
    if extension.lower() not in valid_image_extensions:
        continue
    image_path_list.append(os.path.join(imageDir, file))
 
#loop through image_path_list to open each image
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





#for imagePath in image_path_list:
 #   image = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
  #  width = 224
  #  height = 224
  #  dim = (width, height)
  #  resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
  # list_of_images = np.expand_dims(resized, axis=0)
  #  results = model.predict(list_of_images)
  #  print(results)
  #  single_result = results[0]
  #  most_likely_class_index = int(np.argmax(single_result))
  #  class_likelihood = single_result[most_likely_class_index]
  #  class_label = class_labels[most_likely_class_index]
  #  print("This is image is a {} - Likelihood: {:2f}".format(class_label, class_likelihood))
  #  plt.imshow(image)
    # Show the plot on the screen
  #  plt.show()





