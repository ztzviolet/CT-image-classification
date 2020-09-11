# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:59:32 2020

@author: leafblade
"""
# already trained model on test set
# test script to preform prediction on test images inside 
# dataset/test/
#   -- image_1.jpg
#   -- image_2.jpg
#   ...

# organize imports
from __future__ import print_function

# keras imports
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
%matplotlib inline


# other imports
from sklearn.linear_model import LogisticRegression
import numpy as np
import os
import json
import pickle
import cv2
import glob
import h5py

# load the user configs
with open('conf/conf_inceptionV3.json') as f:    
	config = json.load(f)

# config variables
model_name 		= config["model"]
weights 		= config["weights"]
include_top 	= config["include_top"]
train_path 		= config["train_path"]
test_path 		= config["test_path"]
features_path 	= config["features_path"]
labels_path 	= config["labels_path"]
test_features_path   = config["test_features_path"]
test_labels_path   = config["test_labels_path"]
vali_size 		= config["vali_size"]
results 		= config["results"]
test_results 	= config["test_results"]
model_path 		= config["model_path"]
seed 			= config["seed"]
classifier_path = config["classifier_path"]

# load the trained logistic regression classifier
print ("[INFO] loading the classifier...")
classifier = pickle.load(open(classifier_path, 'rb'))


# get all the train labels
train_labels = os.listdir(train_path)

# import features and labels
h5f_test_data  = h5py.File(test_features_path, 'r')
h5f_test_label = h5py.File(test_labels_path, 'r')

test_features_string = h5f_test_data['dataset_1']
test_labels_string   = h5f_test_label['dataset_1']

test_features = np.array(test_features_string)
test_labels   = np.array(test_labels_string)

h5f_test_data.close()
h5f_test_label.close()

print("[INFO] evaluating model on new test data...")
f = open(test_results, "w")
rank_1 = 0
for (label, features) in zip(test_labels, test_features):
  # predict the probability of each class label and
  # take the top-1 class labels
  predictions = classifier.predict_proba(np.atleast_2d(features))[0]
  predictions = np.argsort(predictions)[::-1]
  # rank-1 prediction increment
  if label == predictions[0]:
    rank_1 += 1

# convert accuracies to percentages
rank_1 = (rank_1 / float(len(test_labels))) * 100

# write the accuracies to file
f.write("Rank-1: {:.2f}%\n".format(rank_1))

# evaluate the model of test data
preds = classifier.predict(test_features)

# write the classification report to file
f.write("{}\n".format(classification_report(test_labels, preds)))
f.close()

# display the confusion matrix
print("[INFO] confusion matrix")

# get the list of training lables
labels = sorted(list(os.listdir(train_path)))

# plot the confusion matrix
fig,ax=plt.subplots()
cm = confusion_matrix(test_labels, preds)
sns.heatmap(cm,
            annot=True,
            cmap="Blues")
ax.set_ylim([0,2])
plt.show()