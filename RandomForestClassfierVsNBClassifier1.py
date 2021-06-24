#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 21:09:35 2021

@author: trombettak
"""

#Dependencies
import pandas as pd
from sklearn.model_selection import train_test_split

# This was imported for the random forest classifier which did much better.
from sklearn.ensemble import RandomForestClassifier




# Read the data in from shared google drive after mounting to drive
with open('/content/drive/Shareddrives/Network Privacy Data/Network Traffic Data.txt') as f:
  lines = [[float(number) if "." in number else int(number) for number in line.strip().split(",")] for line in f.readlines() if line.strip() and not line.startswith('@')]
cols = ["dofM", "dofW", "weekend", "hofD", "mofH", "time", "traffic0", "traffic1", "delta1", "traffic2", "delta2", "traffic3", "delta3", "traffic4", "delta4", "class"]




# Create pandas data frame from data and label columns
df = pd.DataFrame(data=lines, columns=cols)

# Verify correct data frame creation
df




# Split the data with 80% for training and 20% for testing
teacher, student = train_test_split(df, test_size=0.2)

# Split the teacher data with 80% for training and 20% for testing
teacher_train, teacher_test = train_test_split(teacher, test_size=0.2)

# Split the student data with 50% for training and 50% for testing
student_train, student_test = train_test_split(student, test_size=0.5)

# Reindex data after train_test_split's random removal of indices in all sets, leaving gaps
teacher_train = teacher_train.reset_index(drop=True)
teacher_test = teacher_test.reset_index(drop=True)
student_train = student_train.reset_index(drop=True)
student_test = student_test.reset_index(drop=True)

if len(student_train.index) != len(student_test.index):
  student_test = student_test[:-1]

# Partition of data is now
# teacher_train = 64%, teacher_test = 16%
# student_train = 10%, student_test = 10%




# Save elements in all data sets' "class" columns to lists for future reference
teacher_train_classes = teacher_train.iloc[:,15].tolist()
teacher_test_classes = teacher_test.iloc[:,15].tolist()
student_train_classes = student_train.iloc[:,15].tolist()
student_test_classes = student_test.iloc[:,15].tolist()

# Drop "class" column from both train and test data
teacher_train = teacher_train.drop(labels='class', axis=1)
teacher_test = teacher_test.drop(labels='class', axis=1)
student_train = student_train.drop(labels='class', axis=1)
student_test = student_test.drop(labels='class', axis=1)




# Train teacher on data using random forest classifier.

teacher_clf = RandomForestClassifier(random_state=0)
teacher_clf.fit(teacher_train, teacher_train_classes)
teacher_rf_predicted_classes = teacher_clf.predict(teacher_test)

teacher_rf_accuracy = sum(1 for x,y in zip(teacher_rf_predicted_classes,teacher_test_classes) if x == y) / len(teacher_rf_predicted_classes)
print(f"Teacher Random Forest Accuracy: {teacher_rf_accuracy}")




# Predict classes of student_train
student_rf_predicted_classes = teacher_clf.predict(student_train)

# Train student model using predicted labels from teacher model
student_clf = RandomForestClassifier(random_state=0)
student_clf.fit(student_train, student_rf_predicted_classes)
student_rf_predicted_classes = student_clf.predict(student_test)

student_rf_accuracy = sum(1 for x,y in zip(student_rf_predicted_classes,student_test_classes) if x == y) / len(student_rf_predicted_classes)
print(f"Student Random Forest Accuracy: {student_rf_accuracy}")