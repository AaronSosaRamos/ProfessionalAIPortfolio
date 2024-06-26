# -*- coding: utf-8 -*-
"""CB001 - Chapter 4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1q0Hh9cWX4WLSBw1QG0JqxHfRITwGAO9b

# Support Vector Machine


*   Used for both regression and classification tasks
*   Not recommended for large datasets

Purpose: Classify the Loan Status
"""

#Mount the google drive connection to our dataset
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

"""Load the dataset"""

import pandas as pd
df = pd.read_csv('/content/drive/My Drive/AI/datasets/loan_data.csv')

df.head()

df.info()

df.drop("Loan_ID", axis=1, inplace=True)
df.head()

df.shape

"""Check the categorical variables"""

print(df["Gender"].value_counts())
print(df["Education"].value_counts())
print(df["Self_Employed"].value_counts())
print(df["Property_Area"].value_counts())
print(df["Loan_Status"].value_counts())

"""Fill Null and NA data"""

df.isnull().sum()

print(df["Dependents"].value_counts())

print(df["Credit_History"].value_counts())

df['Gender'].fillna('Female', inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna('Yes', inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
df.isnull().sum()

"""Visualize the data

Categorical variables
"""

import seaborn as sns
import matplotlib.pyplot as plt

categorical_vars = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for var in categorical_vars:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=var)
    plt.title(f'Distribution of {var}')
    plt.xlabel(var)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

""" Numeric variables"""

numeric_vars = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

for var in numeric_vars:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[var], kde=True)
    plt.title(f'Distribution of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, y=var)
    plt.title(f'Box plot of {var}')
    plt.ylabel(var)
    plt.show()

"""Encode categorical variables"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

onehot_cols = ['Gender', 'Education', 'Self_Employed', 'Married']
ordinal_cols = ['Property_Area', 'Loan_Status']

onehot_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

ordinal_pipeline = Pipeline([
    ('ordinal', OrdinalEncoder())
])

preprocessor = ColumnTransformer(transformers=[
    ('onehot', onehot_pipeline, onehot_cols),
    ('ordinal', ordinal_pipeline, ordinal_cols)
], remainder='passthrough')

processed_data = preprocessor.fit_transform(df)

encoded_onehot_cols = preprocessor.named_transformers_['onehot'].named_steps['onehot'] \
                      .get_feature_names_out(input_features=onehot_cols)

encoded_cols = list(encoded_onehot_cols) + ordinal_cols + list(df.columns.drop(onehot_cols + ordinal_cols))

processed_df = pd.DataFrame(processed_data, columns=encoded_cols)
print(processed_df)

processed_df.info()

processed_df['Dependents'] = processed_df['Dependents'].str.replace('+', '')
processed_df = processed_df.astype(float)
processed_df['Dependents'] = processed_df['Dependents'].astype(int)

processed_df.info()

processed_df.head()

unique_property_area = df["Property_Area"].unique()

comparison_df = pd.DataFrame({"Old_Property_Area": unique_property_area,
                              "Encoded_Property_Area": processed_df["Property_Area"].unique()})

comparison_df = comparison_df.sort_values(by="Old_Property_Area").reset_index(drop=True)

print(comparison_df)

unique_loan_status = df["Loan_Status"].unique()

comparison_df = pd.DataFrame({"Old_Loan_Status": unique_loan_status,
                              "Encoded_Loan_Status": processed_df["Loan_Status"].unique()})

comparison_df = comparison_df.sort_values(by="Old_Loan_Status").reset_index(drop=True)

print(comparison_df)

"""Use the classification model"""

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, make_scorer

X = processed_df.drop(columns=['Loan_Status'])
y = processed_df['Loan_Status']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

svm_model = SVC(kernel='linear')

svm_model.fit(X_train, y_train)

"""Loss Function => Accuracy"""

val_accuracy = accuracy_score(y_val, svm_model.predict(X_val))
print("Validation Accuracy:", val_accuracy)

test_accuracy = accuracy_score(y_test, svm_model.predict(X_test))
print("Test Accuracy:", test_accuracy)

"""Prediction"""

df_for_test = {
    'Gender_Female': 1.0,
    'Gender_Male': 0.0,
    'Education_Graduate': 1.0,
    'Education_Not Graduate': 0.0,
    'Self_Employed_No': 1.0,
    'Self_Employed_Yes': 0.0,
    'Married_No': 1.0,
    'Married_Yes': 0.0,
    'Property_Area': 2.0,
    'Dependents': 2,
    'ApplicantIncome': 5827.3,
    'CoapplicantIncome': 0.0,
    'LoanAmount': 130.5,
    'Loan_Amount_Term': 352.4,
    'Credit_History': 1.0
}

df_for_test = pd.DataFrame([df_for_test])
predict = svm_model.predict(df_for_test)
print(f"Loan Status prediction: {predict}")