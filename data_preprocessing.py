# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:10:35 2020

@author: leafblade
"""
#split original dataset into train/validation/test，only execute once
import numpy as np # linear algebra
import random
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cv2
import shutil
from glob import glob
# Helper libraries
import matplotlib.pyplot as plt
import math
%matplotlib inline
print(tf.__version__)

path_positive_cases = os.path.join(os.getcwd(),'covidct','CT_COVID')
path_negative_cases = os.path.join(os.getcwd(),'covidct','CT_NonCOVID')

positive_images_ls = glob(os.path.join(path_positive_cases,"*.png"))
negative_images_ls = glob(os.path.join(path_negative_cases,"*.png"))
negative_images_ls.extend(glob(os.path.join(path_negative_cases,"*.jpg")))

covid = {'class': 'CT_COVID',
         'path': path_positive_cases,
         'images': positive_images_ls}

non_covid = {'class': 'CT_NonCOVID',
             'path': path_negative_cases,
             'images': negative_images_ls}

total_positive_covid = len(positive_images_ls)
total_negative_covid = len(negative_images_ls)
print("Total Positive Cases Covid19 images: {}".format(total_positive_covid))
print("Total Negative Cases Covid19 images: {}".format(total_negative_covid))

image_positive = cv2.imread(os.path.join(positive_images_ls[1]))
image_negative = cv2.imread(os.path.join(negative_images_ls[5]))
f = plt.figure(figsize=(8, 8))
f.add_subplot(1, 2, 1)
plt.imshow(image_negative)
f.add_subplot(1,2, 2)
plt.imshow(image_positive)

subdirs  = ['train/','vali/', 'test/']
for subdir in subdirs:
    labeldirs = ['CT_COVID', 'CT_NonCOVID']
    for labldir in labeldirs:
        newdir = subdir + labldir
        os.makedirs(newdir, exist_ok=True)
        
random.seed(123)
vali_ratio=0.15
test_ratio = 0.15
#random.shuffle(positive_images_ls)
#random.shuffle(negative_images_ls)
for cases in [covid, non_covid]:
    total_cases = len(cases['images']) #number of total images
    num_to_select_test = int(test_ratio * total_cases)
    num_to_select_vali = int(vali_ratio * total_cases)
    print(cases['class'],'test set:', num_to_select_test)
    print(cases['class'],'validation set:', num_to_select_vali)
    random.shuffle(cases['images'])
    list_of_test_set=cases['images'][:num_to_select_test]
    list_of_vali_set=cases['images'][num_to_select_test:num_to_select_test+num_to_select_vali]
    list_of_train_set=cases['images'][num_to_select_test+num_to_select_vali:]
    for files in list_of_test_set:
        shutil.copy2(files, 'test/' + cases['class'])
    for files in list_of_vali_set:
        shutil.copy2(files, 'vali/' + cases['class'])
    for files in list_of_train_set:
        shutil.copy2(files, 'train/' + cases['class'])
#for cases in [covid, non_covid]:
#    total_cases = len(cases['images']) #number of total images
#    num_to_select = int(test_ratio * total_cases) #number of images to copy to test set
#    
#    print(cases['class'], num_to_select)
#    
#    list_of_random_files = random.sample(cases['images'], num_to_select) #random files selected
#
#    for files in list_of_random_files:
#        shutil.copy2(files, 'test/' + cases['class'])
#        
## Copy Images to train set
#for cases in [covid, non_covid]:
#    image_test_files = os.listdir('test/' + cases['class']) # list test files 
#    for images in cases['images']:
#        if images.split("\\")[-1] not in (image_test_files): #exclude test files from shutil.copy
#            shutil.copy2(images, 'train/' + cases['class'])
            
total_train_covid = len(os.listdir('./train/CT_COVID'))
total_train_noncovid = len(os.listdir('./train/CT_NonCOVID'))
total_vali_covid = len(os.listdir('./vali/CT_COVID'))
total_vali_noncovid = len(os.listdir('./vali/CT_NonCOVID'))
total_test_covid = len(os.listdir('./test/CT_COVID'))
total_test_noncovid = len(os.listdir('./test/CT_NonCOVID'))

print("Train sets images COVID: {}".format(total_train_covid))
print("Train sets images Non COVID: {}".format(total_train_noncovid))
print("Validation sets images COVID: {}".format(total_vali_covid))
print("Validation sets images Non COVID: {}".format(total_vali_noncovid))
print("Test sets images COVID: {}".format(total_test_covid))
print("Test sets images Non COVID: {}".format(total_test_noncovid))