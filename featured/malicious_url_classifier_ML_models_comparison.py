#-----------------------------------
# GLOBAL FEATURE EXTRACTION
#-----------------------------------

# organize imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import csv
import cv2
import os
import h5py
import os
import glob
import cv2
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
from collections import Counter
import math
from sklearn.feature_extraction.text import CountVectorizer

# filter all the warnings
import warnings

warnings.filterwarnings('ignore')

"""---------------------------------------open data file-------------------------------------------"""

#number of columns in the csv data file
maximum_number_of_feature = 32 #Number of attributes

#file_name of csv data file
file_name = "url_features_extracted_url_column_removed.csv"

#initialise empty list
url_features = []
labels = []

#open csv file
with open(file_name, "r", encoding="utf8") as file:

    #read csv file
    data_file = csv.reader(file, delimiter=',')

    # This skips the first row of the CSV file.
    next(data_file)

    #create url_features and labels lists
    for line in data_file:

        url_features.append(line[0:maximum_number_of_feature])
        labels += line[maximum_number_of_feature]

#convert input data to numpy array
x = np.array(url_features).astype(float)
y = np.array(labels).astype(float)

# verify the shape of the feature vector and labels
print("original data information")
print("features shape: {}".format(x.shape))
print("labels shape: {}".format(y.shape))
print("start_training")


"""------------------------model parameters------------------------------------"""
seed = 9
test_size = 0.10
num_trees = 100

"""------------------------create model----------------------------------------"""
# create all the machine learning models
models = []
models.append(('LR', LogisticRegression(random_state=9)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT'
               '', DecisionTreeClassifier(random_state=9)))
models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=9)))
models.append(('NB', GaussianNB()))

# variables to hold the results and names
results = []
model_name = []
scoring = "accuracy"

"""------------------split the training and testing data----------------------"""
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

print("splitted data information")
print("Train data  : {}".format(x_train.shape))
print("Test data   : {}".format(x_test.shape))
print("Train labels: {}".format(y_train.shape))
print("Test labels : {}".format(y_test.shape))


"""-----------------Training Performance----------------------"""

print("Training Performance:")

# 10-fold cross validation
for name, model in models:
    number_of_fold = KFold(n_splits=10, random_state=7)
    cross_validation_results= cross_val_score(model, x_train, y_train, cv=number_of_fold, scoring=scoring)
    results.append(cross_validation_results)
    model_name.append(name)
    print("%s: %f (%f)" % (name, cross_validation_results.mean(), cross_validation_results.std()))

# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Machine Learning algorithm comparison')
axis = fig.add_subplot(111)
pyplot.boxplot(results)
axis.set_xticklabels(model_name)
pyplot.show()

"""-----------------Training Performance----------------------"""

print("Testing Performance:")

for name, model in models:

    number_of_fold = KFold(n_splits=10, random_state=7)
    cross_validation_results= cross_val_score(model, x_test, y_test, cv=number_of_fold, scoring=scoring)
    results.append(cross_validation_results)
    model_name.append(name)
    print("%s: %f (%f)" % (name, cross_validation_results.mean(), cross_validation_results.std()))