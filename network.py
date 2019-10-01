from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from imutils import paths
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras import regularizers

import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

import random
import cv2
import os
import re  

path = 'the path to the training dataset/'
path_1 = 'the path to the training dataset/'
dir_catalog = os.listdir(path_1)
dir_count = os.listdir(path)

print('---------обработка файлов---------------')
data = []
labels = []
dat = []

for j in range(len(dir_catalog)):
	catalog = os.chdir(path_1 + dir_count[j])
	d = dir_count[j]
	f = os.listdir()
	
	for i in range(len(f)):
		file = f[i]
		image = cv2.imread(file)
		try:
			image = cv2.resize(image,(28,28))
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		except Exception as e:
			print(str(e))
		data.append(np.array(image))
		labels.append(d)
		




labels = np.array(labels)


(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)
trainX = np.array(trainX)
testX = np.array(testX)

trainX = trainX.reshape((31152,28,28,1))
trainX = trainX.astype('float32') / 255
testX = testX.reshape(10384,28,28,1)
testX = testX.astype('float32') / 255



trainY = to_categorical(trainY)
testY = to_categorical(testY)

model = Sequential()

model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(33, activation='softmax'))

#model.add(Dense(len(lb.classes_), activation="softmax"))


adam = Adam(lr=0.001, beta_1=0.7, beta_2=0.888, epsilon=None, decay=0.001, amsgrad=False)

model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(trainX, trainY, validation_data=(testX, testY),epochs=5, batch_size=64)

# функция сохранения модели для андроид устройств

# def export_model_for_mobile(model_name, input_node_name, output_node_name):
#     tf.train.write_graph(K.get_session().graph_def, 'out', \
#         model_name + '_graph.pbtxt')

#     tf.train.Saver().save(K.get_session(), 'out/' + model_name + '.chkp')

#     freeze_graph.freeze_graph('out/' + model_name + '_graph.pbtxt', None, \
#         False, 'out/' + model_name + '.chkp', output_node_name, \
#         "save/restore_all", "save/Const:0", \
#         'out/frozen_' + model_name + '.pb', True, "")

#     input_graph_def = tf.GraphDef()
#     with tf.gfile.Open('out/frozen_' + model_name + '.pb', "rb") as f:
#         input_graph_def.ParseFromString(f.read())

#     output_graph_def = optimize_for_inference_lib.optimize_for_inference(
#             input_graph_def, [input_node_name], [output_node_name],
#             tf.float32.as_datatype_enum)

#     with tf.gfile.FastGFile('path to save folder' + model_name + '.pb', "wb") as f:
#     	f.write(output_graph_def.SerializeToString())

# сохранение модели 
model.save('C:/Users/DDR/Documents/программирование/python2/keras/m_network/model_alphabet.h5')
export_model_for_mobile('xor_nn', "conv2d_1_input", "dense_2/Softmax")

test_loss, test_acc = model.evaluate(testX, testY)
print('точность сети:', test_acc)