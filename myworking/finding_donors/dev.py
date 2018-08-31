# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 15:31:12 2018

@author: derekh
"""

import numpy as np
import pandas as pd
from time import time
from IPython.display import display 
import matplotlib as plt

# Import supplementary visualization code visuals.py
import visuals as vs

# Pretty display for notebooks
#%matplotlib inline

# Load the Census dataset
data = pd.read_csv("census.csv")

# Success - Display the first record
#display(data.head(20))

#1.Data Exploration
# TODO: Total number of records
n_records = len(data)

# TODO: Number of records where individual's income is more than $50,000
n_greater_50k = len(data[data.income == ">50K"])

# TODO: Number of records where individual's income is at most $50,000
n_at_most_50k = len(data[data.income == "<=50K"])

# TODO: Percentage of individuals whose income is more than $50,000
greater_percent = n_greater_50k / n_records * 100

# Print the results
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))

#2.Prepare the Data - Preprocessing
"""
Before data can be used as input for machine learning algorithms, 
it often must be cleaned, formatted, and restructured — this is typically known as preprocessing. 
Fortunately, for this dataset, there are no invalid or missing entries we must deal with, 
however, there are some qualities about certain features that must be adjusted. 
This preprocessing can help tremendously with the outcome and predictive power of nearly all learning algorithms.
"""
# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Visualize skewed continuous features of original data
vs.distribution(data)

# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
print("features_log_transformed:")
display(features_log_transformed.head(20))
#features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
#vs.distribution(features_log_transformed, transformed = True)


