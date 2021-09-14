import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt


data = pd.read_csv('Advertising.csv', index_col=0)
data.head()
data.columns = ['TV', 'Radio', 'Newspaper', 'Sales']
print(data.shape)
fig,axes = plt.subplots(1,3,sharey=True)
data.plot(kind='scatter',x='TV',y='Sales',ax=axes[0],figsize=(16,8))
data.plot(kind='scatter',x='Radio',y='Sales',ax=axes[1])
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axes[2])
feature_cols = ['TV']
x = data[feature_cols]
y = data.Sales

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x,y)

print(lm.intercept_)
print(lm.coef_)

X_new = pd.DataFrame({'TV':[50]})
X_new.head()
lm.predict(X_new)
X_new = pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
X_new.head()

preds = lm.predict(X_new)
print(preds)
data.plot(kind='scatter',x='TV',y='Sales')
plt.plot(X_new,preds,c='red',linewidth=2)

import statsmodels.formula.api as smf
lm = smf.ols(formula='Sales ~ TV',data=data).fit()
lm.conf_int()
lm.pvalues
lm.rsquared

feature_cols = ['TV','Radio','Newspaper']
X = data[feature_cols]
y = data.Sales

from sklearn import model_selection
xtrain,xtest,ytrain,ytest = model_selection.train_test_split(X,y,test_size=0.3,random_state=42)
lm = LinearRegression()
lm.fit(X,y)
print(lm.intercept_)
print(lm.coef_)

lm = LinearRegression()
lm.fit(xtrain,ytrain)
print(lm.intercept_)
print(lm.coef_)
predictions = lm.predict(xtest)
print(sqrt(mean_squared_error(ytest,predictions)))
lm = smf.ols(formula='Sales~TV + Radio + Newspaper', data=data).fit()
lm.conf_int()
lm.summary()

import numpy as np

nums = np.random.seed(12345)
mask_large = nums > 0.5
data['Size'] = 'small'
data.loc[mask_large, 'Size']='large'
data.head()
data['IsLarge'] = data.Szie.map({'small':0,'large':1})
data.head()
feature_cols = ['TV','Radio','Newspaper','IsLarge']
X = data[feature_cols]
y = data.Sales

lm = LinearRegression()
lm.fit(X,y)
zip(feature_cols,lm.coef_)

np.random.seed(123456)
nums = np.random.rand(len(data))
mask_suburban = (nums > 0.33) & (nums < 0.66)
mask_urban = nums > 0.66
data['Area']='rural'
data.loc[mask_suburban,'Area']='suburban'
data.loc[mask_urban,'Area']='urban'
data.head()

# area_dummies = pd.get_dummies(data.Area, prefix = 'Area').iloc[:,1:]
# data=pd.concat([data,area_dummies],axis=1)
data.head()

feature_cols = ['TV', 'Radio', 'Newspaper', 'IsLarge', 'Area_suburban', 'Area_urban']
X = data[feature_cols]
y = data.Sales

lm = LinearRegression()
lm.fit(X,y)
print(feature_cols,lm.coef_)

