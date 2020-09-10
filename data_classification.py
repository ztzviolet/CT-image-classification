# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:05:39 2020

@author: leafblade
"""
#simple sequential CNN, following tut.py
#
import numpy as np # linear algebra
import random
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import cv2
import shutil
from glob import glob
# Helper libraries
import matplotlib.pyplot as plt
import math
%matplotlib inline
print(tf.__version__)

batch_size = 100
epochs = 20
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
vali_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
test_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our test data

train_dir = os.path.join(os.getcwd(),'train')
vali_dir = os.path.join(os.getcwd(),'vali')
test_dir = os.path.join(os.getcwd(),'test')

total_train_covid = len(os.listdir('./train/CT_COVID'))
total_train_noncovid = len(os.listdir('./train/CT_NonCOVID'))
total_vali_covid = len(os.listdir('./vali/CT_COVID'))
total_vali_noncovid = len(os.listdir('./vali/CT_NonCOVID'))
total_test_covid = len(os.listdir('./test/CT_COVID'))
total_test_noncovid = len(os.listdir('./test/CT_NonCOVID'))
total_vali=total_vali_covid + total_vali_noncovid
total_train = total_train_covid + total_train_noncovid
total_test = total_test_covid + total_test_noncovid

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')
vali_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=vali_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')
test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=test_dir,
                                                              shuffle=False,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

model = Sequential([
    Conv2D(32, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1),
])
#model.add(Activation('sigmoid'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()
	
es = EarlyStopping(monitor='val_loss', patience=3,verbose=1)
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=vali_data_gen,
    validation_steps=total_vali // batch_size,
    callbacks=[es]
)

acc = history.history['acc']
val_acc = history.history['val_acc']

loss=history.history['loss']
val_loss=history.history['val_loss']
epochs_range = list(range(len(acc)))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.show()


test_data_gen.reset()
y_pred=model.predict_generator(test_data_gen)
y_pred = 1/(1+np.exp(-y_pred))
y_pred=(y_pred>0.5)
y_pred =y_pred.astype(int)
print('Confusion Matrix')
print(confusion_matrix(vali_data_gen.classes, y_pred))

#83.8