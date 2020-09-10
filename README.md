# CT-image-classification
Spring 2020 EE380L data mining final project

Use CNN to extract features from CT-image of lungs and classify COVID-19 vs Non COVID-19

dataset: https://www.kaggle.com/luisblanche/covidct

data_preprocessing.py: split the image data into training/validation/test sets; only executed once

Folder looks like:
-data_classification.py
-data_preprocessing.py
-train
-----CT_COVID
---------245 images
-----CT_NonCOVID
---------279 images
-vali
-----CT_COVID
---------52 images
-----CT_NonCOVID
---------59 images
-test
-----CT_COVID
---------52 images
-----CT_NonCOVID
---------59 images
