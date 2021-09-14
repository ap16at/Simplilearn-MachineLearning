## Background Problem Statement
# The US Census Bureau has published California Census Data which has 10 types of metrics such as the population,
#  median income, median housing price, and so on for each block group in California. 
# The dataset also serves as an input for project scoping and tries to specify the functional and nonfunctional requirements for it.

## Problem Objective
# The project aims at building a model of housing prices to predict median house values in California using the provided dataset. 
# This model should learn from the data and be able to predict the median housing price in any district, given all the other metrics.

# Districts or block groups are the smallest geographical units for which the US Census Bureau
# publishes sample data (a block group typically has a population of 600 to 3,000 people). There are 20,640 districts in the project dataset.

## Analysis Tasks to be performed:
# 1. Build a model of housing prices to predict median house values in California using the provided dataset.
# 2. Train the model to learn from the data to predict the median housing price in any district, given all the other metrics.
# 3. Predict housing prices based on median_income and plot the regression chart for it.

# importing the necessary libraries
import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor
import statsmodels.formula.api as smf

from sklearn.metrics import mean_squared_error,r2_score
from math import sqrt

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

# 1. Load the data:
df = pd.read_excel("cali_housing.xlsx")
print(df.head())

print(df.columns)

# 2. Handle missing values:
print(df.isnull().sum())

### 207 null values found in 'total_bedrooms' -> replace them with the mean
df.total_bedrooms = df.total_bedrooms.fillna(df.total_bedrooms.mean())
print(df.isnull().sum())

# 3. Encode categroical data:
le = LabelEncoder()
df['ocean_proximity'] = le.fit_transform(df['ocean_proximity'])

# 4. Split the dataset:
# 5. Standardize data:
# 6. Perform Linear Regression:
# 7. Perform Decision Tree Regression:
# 8. PerformRandom Forest Regression:
# 9. Perform Linear Regression with one independent variable :