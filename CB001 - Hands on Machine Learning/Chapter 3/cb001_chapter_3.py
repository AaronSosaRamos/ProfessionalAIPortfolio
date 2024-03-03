# -*- coding: utf-8 -*-
"""CB001 - Chapter 3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1u66g3BxP29ZZd6Kz9vndOaKt0lalbir_

# Linear Regression

Connect to Google Drive:
"""

#Mount the google drive connection to our dataset
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

"""Load the dataset:"""

import pandas as pd
df = pd.read_csv('/content/drive/My Drive/AI/datasets/urban_bliss_index_2024.csv')

df.head()

df.tail(6)

df.info()

df.describe()

import matplotlib.pyplot as plt

hist = df.hist(bins=20)
plt.tight_layout()
plt.show()

df["City"].unique()

df.drop("City", axis=1, inplace=True)

df

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder_ct = ColumnTransformer(
    transformers=[
        ("ordinal_encoder", OrdinalEncoder(), ["Month", "Traffic_Density"])
    ],
    remainder="passthrough"
)

encoded_df = pd.DataFrame(ordinal_encoder_ct.fit_transform(df),
                          columns=ordinal_encoder_ct.get_feature_names_out())

encoded_df.head()

encoded_df["ordinal_encoder__Month"].value_counts()

unique_months = df["Month"].unique()

comparison_df = pd.DataFrame({"Old_Month": unique_months,
                              "Encoded_Month": encoded_df["ordinal_encoder__Month"].unique()})

comparison_df = comparison_df.sort_values(by="Old_Month").reset_index(drop=True)

print(comparison_df)

unique_td = df["Traffic_Density"].unique()

comparison_df = pd.DataFrame({"Old_Traffic_Density": unique_td,
                              "Encoded_Traffic_Density": encoded_df["ordinal_encoder__Traffic_Density"].unique()})

comparison_df = comparison_df.sort_values(by="Old_Traffic_Density").reset_index(drop=True)

print(comparison_df)

encoded_df.info()

"""# Only for an Individual Regression: Happiness_Score"""

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = encoded_df.drop('remainder__Happiness_Score', axis=1)
y = encoded_df['remainder__Happiness_Score']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)

"""Loss Function => MSE"""

from sklearn.metrics import mean_squared_error

val_predictions = linear_regression_model.predict(X_val)
val_mse = mean_squared_error(y_val, val_predictions)
print("Validation MSE:", val_mse)

test_predictions = linear_regression_model.predict(X_test)
test_mse = mean_squared_error(y_test, test_predictions)
print("Test MSE:", test_mse)

cv_scores = cross_val_score(linear_regression_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_mse = -np.mean(cv_scores)
print("Cross-Validation MSE:", cv_mse)

values={
    'ordinal_encoder__Month': 5.0,
    'ordinal_encoder__Traffic_Density': 2.0,
    'remainder__Year': 2025.0,
    'remainder__Decibel_Level': 82.4,
    'remainder__Green_Space_Area': 78.6,
    'remainder__Air_Quality_Index': 56.9,
    'remainder__Cost_of_Living_Index': 85.3,
    'remainder__Healthcare_Index': 89.5
}

df_for_test = pd.DataFrame([values])
predictions = linear_regression_model.predict(df_for_test)
print(f"Happiness Score: {predictions[0]}")

"""#Multiple Regression:


*   Air_Quality_Index
*   Happiness_Score
*   Cost_of_Living_Index
*   Healthcare_Index




"""

X = encoded_df.drop(['remainder__Air_Quality_Index','remainder__Happiness_Score','remainder__Cost_of_Living_Index','remainder__Healthcare_Index'], axis=1)
y = encoded_df[['remainder__Air_Quality_Index','remainder__Happiness_Score','remainder__Cost_of_Living_Index','remainder__Healthcare_Index']]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

from sklearn.multioutput import MultiOutputRegressor

multiple_regression_model = MultiOutputRegressor(LinearRegression())

multiple_regression_model.fit(X_train, y_train)

val_predictions = multiple_regression_model.predict(X_val)
val_mse = mean_squared_error(y_val, val_predictions)
print("Validation MSE:", val_mse)

test_predictions = multiple_regression_model.predict(X_test)
test_mse = mean_squared_error(y_test, test_predictions)
print("Test MSE:", test_mse)

cv_scores = cross_val_score(multiple_regression_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_mse = -np.mean(cv_scores)
print("Cross-Validation MSE:", cv_mse)

values={
    'ordinal_encoder__Month': 5.0,
    'ordinal_encoder__Traffic_Density': 2.0,
    'remainder__Year': 2025.0,
    'remainder__Decibel_Level': 82.4,
    'remainder__Green_Space_Area': 78.6
}

df_for_test = pd.DataFrame([values])
predictions = multiple_regression_model.predict(df_for_test)
print(predictions)