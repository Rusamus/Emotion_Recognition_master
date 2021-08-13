from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from numpy import array
from numpy import argmax
from keras.utils import to_categorical
import pandas as pd


#preprocessing data
!unzip archive.zip
raw_data_csv_file_name = '/content/fer2013.csv'
raw_data = pd.read_csv(raw_data_csv_file_name)
raw_data.info()
raw_data.head()

tok = []
for i in range(raw_data.shape[0]):
  img = raw_data["pixels"][i] #image
  val = img.split(" ")
  x_pixels = np.array(val, 'float32')
  x_pixels /= 255
  x_reshaped = x_pixels.reshape(48,48)
  tok.append(x_reshaped)
ar = np.stack(tok, axis=0)
ar = np.expand_dims(ar, axis=3)
pictures = np.concatenate((ar,ar,ar), axis=3)

#Model
num_classes = 7
topLayerModel = Sequential()
topLayerModel.add(layers.Dense(256, input_shape=(512,), activation='tanh'))
topLayerModel.add(layers.Dropout(0.5))
topLayerModel.add(layers.Dense(num_classes, input_shape=(256,), activation='softmax'))

topLayerModel.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Using vgg16 as transfer learning
vgg16 = VGG16(include_top=False, input_shape=(48, 48, 3), weights='imagenet')

#fitting
batch = 100
y_train = raw_data["emotion"][:]
y_train = array(y_train)
y_train = to_categorical(y_train)

y_test = raw_data["emotion"][:]
y_test = array(y_test)
y_test = to_categorical(y_test)

x_train = np.squeeze(vgg16(np.squeeze(pictures[:,:,:,:], axis = 0)))
x_test = np.squeeze(vgg16(np.squeeze(pictures[:,:,:,:], axis = 0)))

topLayerModel.fit(x_train,y_train,
                    validation_data=(x_test,y_test), 
                    epochs = 200, batch_size=batch)       
       
#prediction                    
model_predictions = topLayerModel.predict(np.squeeze(vgg16(pictures[0:1000,:,:,:])))
                   
