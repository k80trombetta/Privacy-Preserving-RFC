#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 21:09:35 2021

@author: trombettak
"""

#Dependencies

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# These were imported for the naive bayes classifier which did not do well on predictions.
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import accuracy_score

# This was imported for the random forest classifier which did much better.
from sklearn.ensemble import RandomForestClassifier




# Read the data in from shared google drive after mounting to drive
with open('Network Traffic Data.txt') as f:
  lines = [[float(number) if "." in number else int(number) for number in line.strip().split(",")] for line in f.readlines() if line.strip() and not line.startswith('@')]
cols = ["dofM", "dofW", "weekend", "hofD", "mofH", "time", "traffic0", "traffic1", "delta1", "traffic2", "delta2", "traffic3", "delta3", "traffic4", "delta4", "class"]




# Create pandas data frame from data and label columns
df = pd.DataFrame(data=lines, columns=cols)

# Split the data with %80 for training and %20 for testing
train, test = train_test_split(df, test_size=0.2)

# Verify correct data frame creation
print(df)




# Reindex both train and test data
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# Verify reindexing of train
print(train)

# Verify reindexing of test
print(test)




# Save elements in both train and test "class" column to lists for future reference
train_classes = train.iloc[:,15].tolist()
test_classes = test.iloc[:,15].tolist()

# Drop "class" column from both train and test data
train = train.drop(labels='class', axis=1)
test = test.drop(labels='class', axis=1)

# Verify "class" column was dropped from train
print(train)

# Verify "class" column was dropped from test
print(test)




# Normalize data to handle negative delta values

scaler = MinMaxScaler()
train_normalized = pd.DataFrame(data=MinMaxScaler().fit_transform(train), columns=cols[:15])
test_normalized = pd.DataFrame(data=MinMaxScaler().fit_transform(test), columns=cols[:15])

# Verify normalized train dataframe
print(train_normalized)

# Verify normalized train dataframe
print(test_normalized)




# This was my first attempt at training the model, using a naive bayes classifier.

# Use a classifier to fit model to training data
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(train_normalized, train_classes)

# Evaluate classifier on the test data
nb_predicted_classes = naive_bayes_classifier.predict(test_normalized)

# print(f"Predicted classes: {nb_predicted_classes}")
# print(f"Test classes: {test_classes}")

nb_accuracy = accuracy_score(y_true=test_classes, y_pred=nb_predicted_classes)
print(f"Naive Bayes Accuracy: {nb_accuracy}\n")

print(plot_confusion_matrix(estimator=naive_bayes_classifier, X=test_normalized, y_true=test_classes, labels=naive_bayes_classifier.classes_))




# This was my second attempt at training the model, using a random forest classifier.

clf = RandomForestClassifier(random_state=0)
clf.fit(train_normalized, train_classes)
rf_predicted_classes = clf.predict(test_normalized)

# print(f"Predicted classes: {rf_predicted_classes}")
# print(f"Test classes: {test_classes}")

rf_accuracy = sum(1 for x,y in zip(rf_predicted_classes,test_classes) if x == y) / len(rf_predicted_classes)
print(f"Random Forest Accuracy: {rf_accuracy}") 










