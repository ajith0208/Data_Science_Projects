# Importing necessary libraries
import numpy as np
import pandas as pd 
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
import pickle

# Reading the data
iris = pd.read_excel("iris.xls")
print(iris.head())
y = iris['Classification']
iris.drop(columns='Classification',inplace=True)
X = iris[['SL', 'SW', 'PL', 'PW']]

# Training the model
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.3)
model = LogisticRegression()
model.fit(x_train,y_train)

pickle.dump(model,open('model.pkl','wb'))
