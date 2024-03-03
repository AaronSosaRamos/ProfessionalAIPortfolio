# -*- coding: utf-8 -*-
"""CB001 - Chapter 9.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EqmpCbE9SGlIrj8zlXxgNIA3l8v0FRER

#Let's get deep into Classification Tasks

Connect to Google Drive
"""

#Mount the google drive connection to our dataset
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

"""Load the dataset"""

import pandas as pd
df = pd.read_csv('/content/drive/My Drive/AI/Project 4/dataset/apple_quality.csv')

df.head()

"""Process the df"""

df.drop(df.tail(1).index, inplace=True)

df.drop("A_id", axis=1, inplace=True)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

column_transformer = ColumnTransformer([
    ('quality_encoder', OneHotEncoder(), ['Quality']),
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessing', column_transformer),
])

df_encoded = pipeline.fit_transform(df)

import pandas as pd

non_encoded_features = list(df.columns[df.columns != 'Quality'])

encoder = pipeline.named_steps['preprocessing'].named_transformers_['quality_encoder']
encoded_feature_names = encoder.get_feature_names_out(['Quality'])

df_encoded_with_names = pd.DataFrame(df_encoded, columns=list(encoded_feature_names) + non_encoded_features)

print(df_encoded_with_names.head())

"""Stochastich Gradient Descent (SGD)"""

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import numpy as np

X = df_encoded_with_names.drop(columns=['Quality_good'])
y = df_encoded_with_names['Quality_good']

"""Solved Data Type problem: https://stackoverflow.com/questions/74081015/valueerror-supported-target-types-are-binary-multiclass-got-unknown"""

y = np.array(y, dtype=float)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

sgd_clf = SGDClassifier(random_state=42)

"""Cross-Validation with StratifiedKFold"""

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(sgd_clf, X_train, y_train, cv=skf, scoring='accuracy')

print("Cross-Validation Mean Accuracy:", np.mean(cv_scores))

sgd_clf.fit(X_train, y_train)

"""Loss Function => Accuracy"""

y_val_pred = sgd_clf.predict(X_val)
accuracy_val = accuracy_score(y_val, y_val_pred)
print("Validation Set Accuracy:", accuracy_val)

y_test_pred = sgd_clf.predict(X_test)
accuracy_test = accuracy_score(y_test, y_test_pred)
print("Test Set Accuracy:", accuracy_test)

"""Confusion Matrix (With precision, recall and F1)"""

from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=skf)

from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_train, y_train_pred)

print("Confusion Matrix:")
print(conf_matrix)

"""Precision, Recall and F1"""

precision = precision_score(y_train, y_train_pred)
recall = recall_score(y_train, y_train_pred)
f1 = f1_score(y_train, y_train_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

y_scores = cross_val_predict(sgd_clf, X_train, y_train, cv=skf, method="decision_function")

precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    plt.grid(True)

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

"""Precision/Recall Tradeoff => decision_function() for managing threshold and achieve better results

Raising the threshold decreases recall and increase precision (In General)
"""

plt.plot(recalls, precisions, "b-", linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.axis([0, 1, 0, 1])
plt.grid(True)
plt.show()

"""ROC

Tradeoff => Recall and False Positives (FPR)
"""

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.grid(True)
    plt.legend(loc="lower right")

plot_roc_curve(fpr, tpr)
plt.show()

"""AUC must be > 0.5 in order to be a good clf"""

from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_train, y_scores)
print("ROC AUC Score:", roc_auc)

"""*   PR is better for rare positive class or caring more about false positives than false negatives
*   ROC is otherwhise

Other clfs, such as RandomForestClassifier, don't have decision_function. Instead, they have "predict_proba"
"""