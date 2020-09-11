# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:43:27 2020

@author: leafblade
"""
#training using LR/RF/SVM
# organize imports
from __future__ import print_function

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import learning_curve

import numpy as np
import h5py
import os
import json
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# load the user configs
with open('conf/conf_inceptionv3.json') as f:    
  config = json.load(f)

# config variables
vali_size     = config["vali_size"]
seed      = config["seed"]
features_path   = config["features_path"]
labels_path   = config["labels_path"]
results     = config["results"]
classifier_path = config["classifier_path"]
train_path    = config["train_path"]
num_classes   = config["num_classes"]
classifier_path = config["classifier_path"]

# import features and labels
h5f_data  = h5py.File(features_path, 'r')
h5f_label = h5py.File(labels_path, 'r')

features_string = h5f_data['dataset_1']
labels_string   = h5f_label['dataset_1']

features = np.array(features_string)
labels   = np.array(labels_string)

h5f_data.close()
h5f_label.close()

# verify the shape of features and labels
print("[INFO] features shape: {}".format(features.shape))
print("[INFO] labels shape: {}".format(labels.shape))

print("[INFO] training started...")
# split the training and testing data

(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(features),
                                                                  np.array(labels),
                                                                  test_size=vali_size,
                                                                  random_state=seed)

print("[INFO] splitted train and test data...")
print("[INFO] train data  : {}".format(trainData.shape))
print("[INFO] test data   : {}".format(testData.shape))
print("[INFO] train labels: {}".format(trainLabels.shape))
print("[INFO] test labels : {}".format(testLabels.shape))

# use logistic regression as the model
print("[INFO] creating model...")

model_LR = LogisticRegression(random_state=seed)
model_RF = RandomForest(n_estimators=100, random_state=seed)
model_SVC = SVC(probability=True,random_state = seed)

model_LR.fit(trainData, trainLabels)
#model_RF.fit(trainData, trainLabels)
#model_SVC.fit(trainData, trainLabels)

testLabels_pre = model_LR.predict(testData)
score_LR = accuracy_score(testLabels, testLabels_pre)
#testLabels_pre = model_RF.predict(testData)
#score_RF = accuracy_score(testLabels, testLabels_pre)
#testLabels_pre = model_SVC.predict(testData)
#score_SVC = accuracy_score(testLabels, testLabels_pre)
#testLabels_pre = model_vote.predict(testData)
#score_vote = accuracy_score(testLabels, testLabels_pre)
#print('Vote accuracy:',score_vote)
print('LR accuracy:',score_LR)
#print('RF accuracy:',score_RF)
#print('SVC accuracy:',score_SVC)

model=model_LR
# dump classifier to file
print("[INFO] saving model...")
pickle.dump(model, open(classifier_path, 'wb'))


"""
----------------------------------------------------------------------------
"""

# use rank-1 and rank-5 predictions
print("[INFO] evaluating model...")
f = open(results, "w")
rank_1 = 0
#rank_5 = 0

# loop over test data
for (label, features) in zip(testLabels, testData):
  # predict the probability of each class label
  predictions = model.predict_proba(np.atleast_2d(features))[0]
  predictions = np.argsort(predictions)[::-1]

  # rank-1 prediction increment
  if label == predictions[0]:
    rank_1 += 1

# convert accuracies to percentages
rank_1 = (rank_1 / float(len(testLabels))) * 100

# write the accuracies to file
f.write("Rank-1: {:.2f}%\n".format(rank_1))

# evaluate the model of test data
preds = model.predict(testData)

# write the classification report to file
f.write("{}\n".format(classification_report(testLabels, preds)))
f.close()

# dump classifier to file
print("[INFO] saving model...")
pickle.dump(model, open(classifier_path, 'wb'))

# display the confusion matrix
print("[INFO] confusion matrix")

# get the list of training lables
labels = sorted(list(os.listdir(train_path)))

# plot the confusion matrix
cm = confusion_matrix(testLabels, preds)
sns.heatmap(cm,
            annot=True,
            cmap="Set2")
plt.show()