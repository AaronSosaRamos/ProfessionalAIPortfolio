# -*- coding: utf-8 -*-
"""CB001 - Chapter 10.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hHrlTZu1dWBoCTE4s6qSkNPL1CrGNr3I

# Multiclass Classification

Connect to Google Drive
"""

#Mount the google drive connection to our dataset
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

"""Load the dataset"""

import pandas as pd
df_train = pd.read_csv('/content/drive/My Drive/AI/datasets/customer_segmentation_train.csv')

df_train.head()

df_train.info()

df_train["Profession"].unique()

df_train.isnull().sum()

df_train.isna().sum()

"""DF management"""

df_train.drop(columns=['ID', 'Var_1'], inplace=True)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

numeric_features = ['Age', 'Work_Experience', 'Family_Size']
cat_features = ['Gender', 'Ever_Married', 'Graduated']
ordinal_features=['Profession', 'Spending_Score']

target_encoder = OrdinalEncoder()
df_train["Segmentation"] = target_encoder.fit_transform(df_train[["Segmentation"]])

X = df_train.drop(columns=['Segmentation'])

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', cat_transformer, cat_features),
        ('ordinal', ordinal_transformer, ordinal_features)
    ])

"""# One vs One strategy"""

ovo_clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', OneVsOneClassifier(SGDClassifier(random_state=42)))
])

y = df_train['Segmentation']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

ovo_clf_pipeline.fit(X_train, y_train)

"""Confusion Matrix"""

y_val_pred = ovo_clf_pipeline.predict(X_val)

conf_matrix_val = confusion_matrix(y_val, y_val_pred)
print("Confusion Matrix for Validation Set:")
print(conf_matrix_val)

print("\nClassification Report for Validation Set:")
print(classification_report(y_val, y_val_pred))

y_test_pred = ovo_clf_pipeline.predict(X_test)

conf_matrix_test = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix for Test Set:")
print(conf_matrix_test)

print("\nClassification Report for Test Set:")
print(classification_report(y_test, y_test_pred))

"""Precision/Recall and F1"""

precision_val = precision_score(y_val, y_val_pred, average='weighted')
recall_val = recall_score(y_val, y_val_pred, average='weighted')
f1_val = f1_score(y_val, y_val_pred, average='weighted')

print("Precision:", precision_val)
print("Recall:", recall_val)
print("F1-score:", f1_val)

precision_test = precision_score(y_test, y_test_pred, average='weighted')
recall_test = recall_score(y_test, y_test_pred, average='weighted')
f1_test = f1_score(y_test, y_test_pred, average='weighted')

print("Precision:", precision_test)
print("Recall:", recall_test)
print("F1-score:", f1_test)

"""# One vs All strategy (One vs Rest)"""

ova_clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', OneVsRestClassifier(SGDClassifier(random_state=42)))
])

ova_clf_pipeline.fit(X_train, y_train)

"""Confusion Matrix"""

y_val_pred = ova_clf_pipeline.predict(X_val)

conf_matrix_val = confusion_matrix(y_val, y_val_pred)
print("Confusion Matrix for Validation Set:")
print(conf_matrix_val)

print("\nClassification Report for Validation Set:")
print(classification_report(y_val, y_val_pred))

y_test_pred = ova_clf_pipeline.predict(X_test)

conf_matrix_test = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix for Test Set:")
print(conf_matrix_test)

print("\nClassification Report for Test Set:")
print(classification_report(y_test, y_test_pred))

"""# Let's try RandomForestClassifier"""

from sklearn.ensemble import RandomForestClassifier

ovo_rfclf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', OneVsOneClassifier(RandomForestClassifier(random_state=42)))
])

ovo_rfclf_pipeline.fit(X_train, y_train)

"""Confusion Matrix"""

y_val_pred = ovo_rfclf_pipeline.predict(X_val)

conf_matrix_val = confusion_matrix(y_val, y_val_pred)
print("Confusion Matrix for Validation Set:")
print(conf_matrix_val)

print("\nClassification Report for Validation Set:")
print(classification_report(y_val, y_val_pred))

ova_rfclf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', OneVsRestClassifier(RandomForestClassifier(random_state=42)))
])

ova_rfclf_pipeline.fit(X_train, y_train)

"""Confusion Matrix"""

y_val_pred = ova_rfclf_pipeline.predict(X_val)

conf_matrix_val = confusion_matrix(y_val, y_val_pred)
print("Confusion Matrix for Validation Set:")
print(conf_matrix_val)

print("\nClassification Report for Validation Set:")
print(classification_report(y_val, y_val_pred))

"""# Let's try Ensemble Models"""

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

base_classifiers = [
    ('rfc', RandomForestClassifier(random_state=42)),
    ('lr', LogisticRegression(random_state=42)),
    ('svc', SVC(random_state=42))
]

pipelines = []
for name, classifier in base_classifiers:
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    pipelines.append((name, pipeline))

ensemble_classifier = VotingClassifier(estimators=pipelines, voting='hard')

from sklearn.model_selection import GridSearchCV

param_grid = {
    'rfc__classifier__n_estimators': [50, 100, 200],
    'svc__classifier__C': [0.1, 1, 10]
}
grid_search = GridSearchCV(ensemble_classifier, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_pipeline = grid_search.best_estimator_

best_pipeline.fit(X_train, y_train)

"""Confusion Matrix"""

y_val_pred = best_pipeline.predict(X_val)

conf_matrix_val = confusion_matrix(y_val, y_val_pred)
print("Confusion Matrix for Validation Set:")
print(conf_matrix_val)

print("\nClassification Report for Validation Set:")
print(classification_report(y_val, y_val_pred))

"""# Binary Classification => The label has only two different values}
# Multi Label Classification => The model predicts more than 1 label
# Multi Output Classification => The model predicts more than 1 label that have to be done with multi label classification
"""