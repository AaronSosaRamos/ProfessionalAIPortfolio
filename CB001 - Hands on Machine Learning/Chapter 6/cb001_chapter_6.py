# -*- coding: utf-8 -*-
"""CB001 - Chapter 6.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14iJeiEWgh0pepV2r9R6emQMJJMwPDOdU

#Neural Networks for Supervised Learning

Connect to Google Drive
"""

#Mount the google drive connection to our dataset
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import pandas as pd
df = pd.read_csv('/content/drive/My Drive/AI/datasets/US Stock Market Dataset.csv')

df.head()

df.info()

df.shape

df.iloc[0]

df.drop(["Unnamed: 0", "Date"], axis=1, inplace=True)
df.iloc[0]

col_change = ["Bitcoin_Price", "Platinum_Price", "Ethereum_Price", "S&P_500_Price", "Nasdaq_100_Price", "Berkshire_Price", "Gold_Price"]
df[col_change] = df[col_change].apply(pd.to_numeric, errors='coerce')

df.info()

df.isnull().sum()

df.isna().sum()

features = df.columns.tolist()

mean_values = df.mean().fillna(0)
mean_values

"""Using an imputer to fulfill NA and null values"""

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

imputer_transformer = SimpleImputer(strategy='constant', fill_value=0)

#There is a bug with strategy = 'mean' which deletes the features with NA values
#Recovered from: https://github.com/scikit-learn/scikit-learn/issues/16426

preprocessor = ColumnTransformer(
    transformers=[
        ('impute', imputer_transformer, features)
    ], remainder='passthrough'
)

imputer_pipeline = Pipeline([
    ('preprocessor', preprocessor)
])

df_imputed = imputer_pipeline.fit_transform(df)

df_imputed = pd.DataFrame(df_imputed, columns=features)

df_imputed

df_imputed.isnull().sum()

df_imputed.isna().sum()

"""Visualize the data"""

import matplotlib.pyplot as plt
import seaborn as sns

df_imputed.hist(figsize=(15, 15))
plt.show()

plt.figure(figsize=(15, 10))
sns.boxplot(data=df_imputed)
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(15, 10))
sns.heatmap(df_imputed.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Meta_Price', y='Natural_Gas_Price', data=df_imputed)
plt.title('Meta_Price vs Natural_Gas_Price')
plt.show()

"""Use a neural network architecture to predict thr Bitcoin Price"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import KFold

X = df_imputed.drop(columns=['Meta_Price'])
y = df_imputed['Meta_Price']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

"""Define the neural network architecture"""

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')
])

"""Compile the model"""

model.compile(optimizer='adam', loss='mean_squared_error')

"""Train the model"""

history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_val_scaled, y_val), verbose=0)

"""Loss Function => MSE"""

val_loss = model.evaluate(X_val_scaled, y_val)
print("Validation Loss:", val_loss)

test_loss = model.evaluate(X_test_scaled, y_test)
print("Test Loss:", test_loss)

y_val_pred = model.predict(X_val_scaled)
y_test_pred = model.predict(X_test_scaled)

val_mse = mean_squared_error(y_val, y_val_pred)
print("Validation MSE:", val_mse)

test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE:", test_mse)

"""Prediction:"""

data_test = {
    'Natural_Gas_Price': 2.178,
    'Natural_Gas_Vol.': 0.0,
    'Crude_oil_Price': 85.6,
    'Crude_oil_Vol.': 3.9512,
    'Copper_Price': 3.8914,
    'Copper_Vol.': 0.0,
    'Bitcoin_Price': 0.0,
    'Bitcoin_Vol.': 46241,
    'Platinum_Price': 976.2,
    'Platinum_Vol.': 0.0,
    'Ethereum_Price': 2523.51,
    'Ethereum_Vol.': 223123,
    'S&P_500_Price': 5421.34,
    'Nasdaq_100_Price': 17521.41,
    'Nasdaq_100_Vol.': 315620000.0,
    'Apple_Price': 196.31,
    'Apple_Vol.': 104350000,
    'Tesla_Price': 194.41,
    'Tesla_Vol.': 123410000,
    'Microsoft_Price': 452.31,
    'Microsoft_Vol.': 29340510,
    'Silver_Price': 24.512,
    'Silver_Vol.': 0.0,
    'Google_Price': 152.51,
    'Google_Vol.': 63700301,
    'Nvidia_Price': 675.8,
    'Nvidia_Vol.': 48350203,
    'Berkshire_Price': 512.212,
    'Berkshire_Vol.': 11350,
    'Netflix_Price': 512.34,
    'Netflix_Vol.': 3960320,
    'Amazon_Price': 185.32,
    'Amazon_Vol.': 137630046,
    'Meta_Vol.': 64710420,
    'Gold_Price': 2512.35,
    'Gold_Vol.': 0.0
}

df_test = pd.DataFrame(data_test, index=[0])

X_test_scaled = scaler.transform(df_test)

predictions = model.predict(X_test_scaled)

y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))

inversed = scaler.inverse_transform(predictions)
print("The Predicted Meta Price is: ",inversed[0][0])

"""# Comparison with other model"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf_regressor = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

best_params

best_rf_regressor = RandomForestRegressor(**best_params, random_state=42)
best_rf_regressor.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score

y_val_pred = best_rf_regressor.predict(X_val)
mse_val = mean_squared_error(y_val, y_val_pred)
print("Mean Squared Error on Validation Set:", mse_val)

y_pred = best_rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on Test Set:", mse)

cv_scores = cross_val_score(best_rf_regressor, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
mse_cv = -cv_scores.mean()
print("Mean Squared Error during Cross-Validation:", mse_cv)

df_test = pd.DataFrame(data_test, index=[0])

prediction = best_rf_regressor.predict(df_test)
print("The Predicted Meta Price is: ",prediction[0])