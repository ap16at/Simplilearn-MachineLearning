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

from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

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
X_Features=['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'ocean_proximity']
X=df[X_Features]
Y=df['median_house_value']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=1)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 5. Standardize data:
col_names = df.columns
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=col_names)
print(scaled_df.head())

# 6. Perform Linear Regression:
linreg = LinearRegression()
linreg.fit(x_train, y_train)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

y_predict = linreg.predict(x_test)

print(sqrt(mean_squared_error(y_test,y_predict)))
print(r2_score(y_test,y_predict))

# 7. Perform Decision Tree Regression:
dtreg = DecisionTreeRegressor()
dtreg.fit(x_train, y_train)
DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None, 
                        max_leaf_nodes=None, min_impurity_decrease=0.0, 
                        min_impurity_split=None, min_samples_leaf=1, 
                        min_samples_split=2, min_weight_fraction_leaf=0.0, 
                        random_state=None, splitter='best')

y_predict = dtreg.predict(x_test)

print(sqrt(mean_squared_error(y_test,y_predict)))
print(r2_score(y_test,y_predict))

# 8. Perform Random Forest Regression:
rfreg = RandomForestRegressor()
rfreg.fit(x_train, y_train)
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                        max_features='auto', max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=1, min_samples_split=2,
                        min_weight_fraction_leaf=0.0, n_estimators=10,
                        n_jobs=None, oob_score=False, random_state=None,
                        verbose=0, warm_start=False)

y_predict = rfreg.predict(x_test)

print(sqrt(mean_squared_error(y_test,y_predict)))
print(r2_score(y_test,y_predict))

# 9. Perform Linear Regression with one independent variable :
x_train_income = x_train[['median_income']]
x_test_income = x_test[['median_income']]

print(x_train_income.shape)
print(y_train.shape)

linreg = LinearRegression()
linreg.fit(x_train_income, y_train)
y_predict = linreg.predict(x_test_income)

print(linreg.intercept_, linreg.coef_)
print(sqrt(mean_squared_error(y_test,y_predict)))
print(r2_score(y_test,y_predict))