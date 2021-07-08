#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 16:00:58 2021

@author: trombettak
"""

#Dependencies
import pandas as pd
import numpy as np


# Functions

# Scoring function for labels

def score(data, option):
  return data.value_counts()[option]/1000


# Read the data in from shared google drive after mounting to drive
with open('/content/drive/Shareddrives/Network Privacy Data/Network Traffic Data.txt') as f:
  lines = [[float(number) if "." in number else int(number) for number in line.strip().split(",")] for line in f.readlines() if line.strip() and not line.startswith('@')]
cols = ["dofM", "dofW", "weekend", "hofD", "mofH", "time", "traffic0", "traffic1", "delta1", "traffic2", "delta2", "traffic3", "delta3", "traffic4", "delta4", "class"]


# Create pandas data frame from data and label columns
df = pd.DataFrame(data=lines, columns=cols)
print(df)


# Get unique class options from class column of df
options = np.sort(df['class'].unique())

epsilon = 1
sensitivity = 1

# Calculate scores for each class label in options
scores = [score(df['class'], option) for option in options]




# Exponential method

# Calculate probability of each class label, based on its score
probabilities = [np.exp(epsilon * score / (2 * sensitivity)) for score in scores]

# Normalize probabilties to sum to 1
probabilities = probabilities / np.linalg.norm(probabilities, ord=1)

# Generate a list of 200 randomly sampled elements from the class labels, based on their probability scores
r_exponential = [np.random.choice(options, 1, p=probabilities)[0] for i in range(200)]

# Print results as a DataFrame
pd.Series(r_exponential).value_counts().rename_axis('class').reset_index(name='most sampled count')





# Report noisy max method (can only be used with finite sets of class labels)

# 200 times, generate a list of 8 laplacian noises for the scores, find the max of those, and add the class label
# associated with the index of the max noise, to r_noisy_max 
r_noisy_max = [options[np.argmax([np.random.laplace(score, sensitivity, epsilon) for score in scores])] for i in range(200)]

# Print results as a DataFrame
pd.Series(r_noisy_max).value_counts().rename_axis('class').reset_index(name='most noisy count')

