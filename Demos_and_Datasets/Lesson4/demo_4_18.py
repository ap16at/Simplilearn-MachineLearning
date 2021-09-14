import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt

train = pd.read_csv('bigmart_train.csv')
# print(train.head(10))
print(train.shape)
print(train.isnull().sum())
print(train['Item_Fat_Content'].unique())
print(train['Outlet_Establishment_Year'].unique())
train['Outlet_Age']=2018-train['Outlet_Establishment_Year']
print(train.head())
print(train['Outlet_Size'].unique())
print(train.describe())
print(train['Item_Fat_Content'].value_counts())
print(train['Outlet_Size'].mode()[0])
train['Outlet_Size']=train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0])
train['Item_Weight']=train['Item_Weight'].fillna(train['Item_Weight'].mean())
train['Item_Visibility'].hist(bins=20)
