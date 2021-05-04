# -*- coding: utf-8 -*-
"""
Created on Sun May  2 23:15:08 2021

@author: SHUBHAM PC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing dataset 

df = pd.read_csv('car price.csv')



df = df.drop(['car_ID','CarName','symboling'],axis=1)


pd.set_option('display.max_columns', None)
print(df.head())



from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

df_col=("fueltype","aspiration","doornumber","carbody","drivewheel","enginelocation","enginetype","cylindernumber","fuelsystem")



for i in df_col:
    df[i]=le.fit_transform(df[i])

print(df.head())

print(df.info())



X= df.iloc[:,:-1].values
y= df.iloc[:,-1].values

print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

y_pred=regressor.predict(X_test)

print(y_pred)





import seaborn as sns
plt.subplot(1,2,1)
plt.title("Car Price Spread")
sns.boxplot(y=df.price)

plt.subplot(1,2,2)
plt.title("Car Price Distribution Plot")
sns.distplot(df.price)
plt.show()


















