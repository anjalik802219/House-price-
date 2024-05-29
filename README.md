# House-price-
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score

%matplotlib inline

df=pd.read_csv("C:\\Users\\Anjali Kumari\\Downloads\\House price\\data.csv")

df.head()

df.info()

df.values

df.columns

df.describe

df["country"].values

df["city"].drop_duplicates().values

df["country"].drop_duplicates().values

df["street"].drop_duplicates().values

df[['yr_built']].drop_duplicates().values.reshape(1,-1)

df['yr_renovated'].drop_duplicates().values.reshape(1,-1)

df['condition'].value_counts(normalize=True)

df['bathrooms'].value_counts(normalize=True)

df['bedrooms'].value_counts(normalize=True)

df['floors'].value_counts(normalize=True)

df['waterfront'].value_counts(normalize=True)


df['view'].value_counts(normalize=True)

df['sqft_basement'].value_counts(normalize=True)

df[['yr_built','yr_renovated']].min()

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the California Housing dataset
california = fetch_california_housing()
X = california.data
y = california.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Optionally, print coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


df[(dataset['yr_built']>2000) & (dataset['view']>0)].shape

df[(dataset['yr_built']<2000) & (dataset['view']>1)].shape

df[(dataset['yr_built']<2000) & (dataset['view']>1)].shape

df[(dataset['yr_built']>2000) & (dataset['floors']>1) & (dataset['waterfront']>0) & (dataset['condition']>3)]

df[(dataset['yr_built']<1999) & (dataset['floors']>1) & (dataset['waterfront']>0) & (dataset['condition']>3)].shape

X=dataset[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view',
           'condition','sqft_above','sqft_basement','yr_built','yr_renovated']]
Y=dataset[['price']]

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=1)

model=LinearRegression()

model.fit(X_train,y_train)

pip install scikit-learn

from sklearn.metrics import mean_squared_error

# Assuming y_pred and y_test are already defined
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


model.coef_

model.intercept_

model.score(X_train,y_train)

model.score(X_test,y_test)

dataset2=dataset.drop(['date','city','street','statezip','country','price'],axis=1)
dataset2.corr()

