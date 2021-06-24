#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 21:09:35 2021

@author: trombettak
"""


#Dependencies
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# Used for finding most predicted labels from multiple parents
from collections import Counter

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




# Class to represent all teachers' and student data, models, and predictions
class ModelData:
  def __init__(self, train, test, train_classes, test_classes):
    self.train = train
    self.test = test
    self.train_classes = train_classes
    self.test_classes = test_classes
    self.model = None
    self.prediction = None
    
    
    
    

# Split the data into train and test for teacher and student data sets
def create_split_data(dataSet, dataSetType, testSize):
  train, test = train_test_split(dataSet, test_size=testSize)
  train = train.reset_index(drop=True)
  test = test.reset_index(drop=True)
  if dataSetType == "Student" and len(train.index) != len(test.index):
    test = test[:-1] 
  train_classes = train.iloc[:,15].tolist()
  test_classes = test.iloc[:,15].tolist()
  train = train.drop(labels='class', axis=1)
  test = test.drop(labels='class', axis=1)
  return ModelData(train, test, train_classes, test_classes)

# Split the data with 80% for teachers and 20% for student
teacher, student = train_test_split(df, test_size=0.2)

# Split the teacher into 4 sub-teachers
t1, t2, t3, t4 = np.array_split(teacher, 4)
teachers = [t1, t2, t3, t4]

# Split the teacher data into 4 subsets, and create ModelData class instance for each teacher
for teacher in range(len(teachers)):
  teachers[teacher] = create_split_data(teachers[teacher], "Teacher", 0.2)

# Split the student data and create ModelData class instance for the student
student = create_split_data(student, "Student", 0.5)

# # Partition of data is now
# # teachers = [ {16%, 4%}, {16%, 4%}, {16%, 4%}, {16%, 4%} ]
# # student = {10%, 10%}




# Train teacher on data using random forest classifier.

for teacher in range(len(teachers)):
  teacher_clf = RandomForestClassifier(random_state=0)
  teacher_clf.fit(teachers[teacher].train, teachers[teacher].train_classes)
  teachers[teacher].model = teacher_clf
  teachers[teacher].prediction = teacher_clf.predict(teachers[teacher].test)
  teacher_rf_accuracy = sum(1 for x,y in zip(teachers[teacher].prediction, teachers[teacher].test_classes) if x == y) / len(teachers[teacher].prediction)
  print(f"\nTeacher #{teacher + 1} Accuracy: {teacher_rf_accuracy}")
  
  
  
  
  # Predict classes of student train data using each teacher

# Holds each 4 parents' predictions on student train data
teacherPredictionsForStudent = []
for teacher in range(len(teachers)):
  teacherPredictionsForStudent.append(teachers[teacher].model.predict(student.train))

# Holds element-wise most predicted labels from 4 parents' predictions on student train data
mostPredicted = []
for prediction in range(len(teacherPredictionsForStudent[0])):
  mode = Counter([teacherPredictionsForStudent[0][prediction], teacherPredictionsForStudent[1][prediction],
            teacherPredictionsForStudent[2][prediction], teacherPredictionsForStudent[3][prediction]]).most_common(1)[0][0]
  mostPredicted.append(mode)

sensitivity = 0.1
epsilon = 0.3
print(mostPredicted)
noise = np.random.laplace()
mostPredicted = [round(x + 1) for x in mostPredicted]
print(noise)
print(mostPredicted)

# Train student model using most predicted labels from teacher models
student_clf = RandomForestClassifier(random_state=0)
student_clf.fit(student.train, mostPredicted)
student_rf_predicted_classes = student_clf.predict(student.test)

student_rf_accuracy = sum(1 for x,y in zip(student_rf_predicted_classes,student.test_classes) if x == y) / len(student_rf_predicted_classes)
print(f"Student Accuracy: {student_rf_accuracy}")





    
    
    
    