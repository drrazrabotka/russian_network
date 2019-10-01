from keras.datasets import mnist
from tensorflow import keras
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np 
import matplotlib.pyplot as plt 
#from skimage import data, io, filters, segmentation
import skimage.color as color
import cv2
import os 

###############загрузка модели#################


model = keras.models.load_model('path to the folder where the model is located/model.h5')

#file = os.listdir(path)

img = cv2.imread('image.jpg')
img = cv2.resize(img,(28,28))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img.reshape(1,28,28,1)
img = img.astype('float32') / 255

preds = model.predict(img)
#############выводим результат###############
print(preds.argmax(axis=1)[0])