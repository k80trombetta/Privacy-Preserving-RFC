#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 21:09:35 2021

@author: trombettak
"""

#Dependencies

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics



# Classes

# Represent each teacher's and the student data, trained models, and predictions
class ModelData:
  def __init__(self, train, test, train_classes, test_classes):
    self.train = train
    self.test = test
    self.train_classes = train_classes
    self.test_classes = test_classes
    self.model = None
    self.prediction = None



# Functions 

# Create and return ModelData object using the dataSet parameter
# Parameters
# dataSet: a dataframe with class labels in last column
# dataSetType: will be "Teacher" or "Student"
# testSize: % of dataSet to save for testing
# noise: noise to be added to train and test class labels

def createModelData(dataSet, dataSetType, testSize, noise):
  # Split dataSet into train and test sets
  train, test = train_test_split(dataSet, test_size=testSize)

  # Force equal size student train and test sets
  if dataSetType == "Student" and len(train.index) != len(test.index):
    test = test[:-1] 

  # Create lists from class labels in train and test datasets
  train_classes = train.iloc[:,15].tolist()
  test_classes = test.iloc[:,15].tolist()

  # Add noise to labels
  train_classes = [x + noise for x in train_classes]
  test_classes = [x + noise for x in test_classes]

  # Remove class labels from train and test set dataframes
  train = train.drop(labels='class', axis=1)
  test = test.drop(labels='class', axis=1)

  return ModelData(train, test, train_classes, test_classes)



# Prints the metrics for results on given test and prediction data
def printMetrics(test, predict):
    print('Mean Squared Error (MSE):', metrics.mean_squared_error(test, predict))
    print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(test, predict)))
    mape = np.mean(np.abs((test - predict) / np.abs(test)))
    print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
    print('Accuracy:', round(100*(1 - mape), 2))
    print()
    
    
    
def main():
    
    # Read the data in from file
    with open('Network Traffic Data.txt') as f:
      lines = [[float(number) if "." in number else int(number) for number in line.strip().split(",")] for line in f.readlines() if line.strip() and not line.startswith('@')]
    cols = ["dofM", "dofW", "weekend", "hofD", "mofH", "time", "traffic0", "traffic1", "delta1", "traffic2", "delta2", "traffic3", "delta3", "traffic4", "delta4", "class"]
    
    # Create pandas data frame from data and label columns
    df = pd.DataFrame(data=lines, columns=cols)
    df
    
    # Initially, split original data set 80% for teachers and 20% for student
    teacher, student = train_test_split(df, test_size=0.2)
    
    # Split teacher into 4 sub-teachers and add all to "teachers" list
    t1, t2, t3, t4 = np.array_split(teacher, 4)
    teachers = [t1, t2, t3, t4]
    
    # Calculate Laplacian noise to be added to labels and verify value
    noise = float(str(round(np.random.laplace(), 10)))
    print("Noise:", noise)
    
    # Create ModelData instance for each teacher in "teachers" and reassign it as value in teachers[x], respectively
    for teacher in range(len(teachers)):
      teachers[teacher] = createModelData(teachers[teacher], "Teacher", 0.2, noise)
    
    # Split the student data and create ModelData instance for the student
    student = createModelData(student, "Student", 0.5, noise)
    
    # Partition of data is now...
    
    # teachers = [ ModelData{train: 16%, test: 4%}, ModelData{train: 16%, test: 4%}, 
    #              ModelData{train: 16%, test: 4%}, ModelData{train: 16%, test: 4%} ] = 80% of orginial data
    
    # student = ModelData{train: 10%, test: 10%} = 20% of original data
    
    
    # Train teacher on data using random forest regressor, predict test labels, and print metrics
    print("TEACHER METRICS\n")
    
    for teacher in range(len(teachers)):
      teacher_clf = RandomForestRegressor(random_state=0)
      teacher_clf.fit(teachers[teacher].train, teachers[teacher].train_classes)
      teachers[teacher].model = teacher_clf
      teachers[teacher].prediction = teacher_clf.predict(teachers[teacher].test)
      print(f"Teacher #{teacher + 1}")
      printMetrics(teachers[teacher].test_classes, teachers[teacher].prediction)
     
    # To hold each 4 parents' predictions on student train data
    teacherPredictionsForStudent = []
    
    # Predict classes of student train using each teacher model and assign to "teacherPredictionsForStudent"
    for teacher in range(len(teachers)):
      teacherPredictionsForStudent.append(teachers[teacher].model.predict(student.train))
    
    # To hold mode between each element in teacherPredictionsForStudent at each index y
    mostPredicted = []
    
    # Add mode between each teacher's predictions to "mostPredicted" to form one prediciton list 
    # We will use "mostPredicted" as true labels for student test data
    
    for prediction in range(len(teacherPredictionsForStudent[0])):
      mode = Counter([teacherPredictionsForStudent[0][prediction], teacherPredictionsForStudent[1][prediction],
                teacherPredictionsForStudent[2][prediction], teacherPredictionsForStudent[3][prediction]]).most_common(1)[0][0]
      mostPredicted.append(mode)
     
    # Train student model using most predicted labels from teachers and print metrics
    student_clf = RandomForestRegressor(random_state=0)
    student_clf.fit(student.train, mostPredicted)
    student_rf_predicted_classes = student_clf.predict(student.test)
    
    print("STUDENT METRICS\n")
    printMetrics(student.test_classes, student_rf_predicted_classes)
    
    
    
main()
