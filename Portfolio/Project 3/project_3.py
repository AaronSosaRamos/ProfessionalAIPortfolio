# -*- coding: utf-8 -*-
"""Project 3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WcjCEsbNBeql-hLovo2OtG966DHWw-98

# Project 3 - Hotel KPIs and Critical Dimensions Models - Wilfredo Aaron Sosa Ramos

# Connect the Project to Google Drive
"""

#Mount the google drive connection to our dataset
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

"""# Phase 1: Extract

# Load our Dataset
"""

import pandas as pd
df = pd.read_csv('/content/drive/My Drive/AI/Project 3/dataset/hotel_booking_data_cleaned.csv')

"""# Check out Dataset"""

df

"""# DataFrame metadata"""

df.info()

"""# Select a random row form out Dataset"""

df.iloc[152]

"""# Dataset statistical info."""

df.describe()

"""# Null values"""

df.isnull().sum()

df.shape

"""# Phase 2: Transform

# Duplicated data
"""

df.duplicated()

df_no_duplicated = df.drop_duplicates()
df_no_duplicated.shape

"""# Empty data

Feature Engineering => Remove unnecessary dimensions
"""

df_empty_data_management = df_no_duplicated.drop(["agent","company"], axis=1)
df_empty_data_management.iloc[157]

print(df_empty_data_management.isnull().sum())
print("Actual DF shape: ",df_empty_data_management.shape)

"""KPIs:


*   Lead Time
*   Cancellation Ratio
*   Required Parking Spaces Utilization vs. Total Special Requests
*   Booking Changes to Waiting Time Ratio
*   Ratio of Adults vs. Children vs. Babies
*   Special Requests per Guest
*   Lead Time vs. Previous Cancellations Ratio
*   Booking Changes per Arrival Date Week Number

Critical Dimensions:


*   Market Segment
*   Distribution Channel

# Feature Engineering (For KPIs)

Cancellation Ratio
"""

df_empty_data_management['cancelation_ratio'] = df_empty_data_management['previous_cancellations'] / (df_empty_data_management['previous_cancellations'] + df_empty_data_management['previous_bookings_not_canceled'])

"""Required Parking Spaces Utilization vs. Total Special Requests"""

df_empty_data_management['parking_special_ratio'] = df_empty_data_management['required_car_parking_spaces'] / df_empty_data_management['total_of_special_requests']

"""Booking Changes to Waiting Time Ratio"""

df_empty_data_management['booking_changes_waiting_time_ratio'] = df_empty_data_management['booking_changes'] / df_empty_data_management['days_in_waiting_list']

"""Ratio of Adults vs. Children vs. Babies"""

df_empty_data_management['adults_ratio'] = df_empty_data_management['adults'] / (df_empty_data_management['adults'] + df_empty_data_management['children'] + df_empty_data_management['babies'])
df_empty_data_management['children_ratio'] = df_empty_data_management['children'] / (df_empty_data_management['adults'] + df_empty_data_management['children'] + df_empty_data_management['babies'])
df_empty_data_management['babies_ratio'] = df_empty_data_management['babies'] / (df_empty_data_management['adults'] + df_empty_data_management['children'] + df_empty_data_management['babies'])

"""Special Requests per Guest"""

df_empty_data_management['special_requests_per_guest'] = df_empty_data_management['total_of_special_requests'] / (df_empty_data_management['adults'] + df_empty_data_management['children'] + df_empty_data_management['babies'])

"""Lead Time vs. Previous Cancellations Ratio"""

df_empty_data_management['lead_time_previous_cancel_ratio'] = df_empty_data_management['lead_time'] / df_empty_data_management['previous_cancellations']

"""Booking Changes per Arrival Date Week Number"""

df_empty_data_management['booking_changes_per_week'] = df_empty_data_management['booking_changes'] / df_empty_data_management['arrival_date_week_number']

df.info()

"""# Statistic Graphics for KPIs

# 1. Lead Time
"""

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.histplot(df_empty_data_management['lead_time'], bins=30, kde=True)
plt.title('Lead Time Distribution')
plt.xlabel('Lead Time')
plt.ylabel('Frequency')
plt.show()

"""# 2. Cancellation Ratio

"""

plt.figure(figsize=(8, 6))
sns.barplot(x=df_empty_data_management['is_canceled'], y=df_empty_data_management['cancelation_ratio'])
plt.title('Cancellation Ratio')
plt.xlabel('Canceled')
plt.ylabel('Cancellation Ratio')
plt.show()

"""# 3. Required Parking Spaces Utilization vs. Total Special Requests"""

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_empty_data_management['required_car_parking_spaces'], y=df_empty_data_management['total_of_special_requests'])
plt.title('Required Parking Spaces Utilization vs. Total Special Requests')
plt.xlabel('Required Parking Spaces')
plt.ylabel('Total Special Requests')
plt.show()

"""# 4. Booking Changes to Waiting Time Ratio"""

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_empty_data_management['booking_changes'], y=df_empty_data_management['days_in_waiting_list'])
plt.title('Booking Changes to Waiting Time Ratio')
plt.xlabel('Booking Changes')
plt.ylabel('Days in Waiting List')
plt.show()

"""# 5. Ratio of Adults vs. Children vs. Babies

"""

adults_children_babies = df_empty_data_management[['adults', 'children', 'babies']].sum()
plt.figure(figsize=(8, 6))
adults_children_babies.plot(kind='bar', stacked=True)
plt.title('Ratio of Adults vs. Children vs. Babies')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()

"""# 6. Special Requests per Guest

"""

plt.figure(figsize=(8, 6))
sns.histplot(df_empty_data_management['special_requests_per_guest'], bins=20, kde=True)
plt.title('Special Requests per Guest Distribution')
plt.xlabel('Special Requests per Guest')
plt.ylabel('Frequency')
plt.show()

"""# 7. Lead Time vs. Previous Cancellations Ratio

"""

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_empty_data_management['lead_time'], y=df_empty_data_management['previous_cancellations'])
plt.title('Lead Time vs. Previous Cancellations Ratio')
plt.xlabel('Lead Time')
plt.ylabel('Previous Cancellations')
plt.show()

"""# 8. Booking Changes per Arrival Date Week Number

"""

plt.figure(figsize=(10, 6))
sns.barplot(x=df_empty_data_management['arrival_date_week_number'], y=df_empty_data_management['booking_changes'])
plt.title('Booking Changes per Arrival Date Week Number')
plt.xlabel('Arrival Date Week Number')
plt.ylabel('Booking Changes')
plt.xticks(rotation=45)
plt.show()

"""# Removing unnecessary dimensions"""

df_empty_data_management.info()

"""Drop unnecessary dimensions"""

columns_to_drop = ['hotel', 'arrival_date_month', 'meal', 'country',
                   'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type',
                   'reservation_status', 'reservation_status_date']

df_cleaned = df_empty_data_management.drop(columns=columns_to_drop)

df_cleaned.info()

"""# Encoding categorical values"""

df_cleaned["market_segment"].value_counts()

df_cleaned["distribution_channel"].value_counts()

categorical_features = df_cleaned.select_dtypes(include=['object']).columns
categorical_features

"""Creating pipelines"""

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
])

onehot_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

numeric_features = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
numeric_features

"""Creating a pre-processor"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', onehot_transformer, categorical_features)
    ])

"""Clean data from inf. and nan values"""

import numpy as np

df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
df_cleaned.dropna()
df_cleaned.shape

"""Develop and use the Pipeline (StandardScaler and OneHot Encoder)"""

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

df_encoded = pipeline.fit_transform(df_cleaned)
df_encoded = pd.DataFrame(df_encoded, columns=pipeline.named_steps['preprocessor'].get_feature_names_out())
df_encoded.info()

"""# Phase 3: Load

# KPIs

Alternative 1: MultiOutputRegressor with RandomForestRegressor
"""

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

X = df_encoded.drop(['num__cancelation_ratio', 'num__parking_special_ratio', 'num__booking_changes_waiting_time_ratio',
               'num__adults_ratio', 'num__children_ratio', 'num__babies_ratio', 'num__special_requests_per_guest',
               'num__lead_time_previous_cancel_ratio', 'num__booking_changes_per_week', 'num__lead_time'], axis=1)
y = df_encoded[['num__cancelation_ratio', 'num__parking_special_ratio', 'num__booking_changes_waiting_time_ratio',
          'num__adults_ratio', 'num__children_ratio', 'num__babies_ratio', 'num__special_requests_per_guest',
          'num__lead_time_previous_cancel_ratio', 'num__booking_changes_per_week', 'num__lead_time']]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

multi_output_regressor = MultiOutputRegressor(RandomForestRegressor(random_state=42))

multi_output_regressor.fit(X_train, y_train)

y_val_pred = multi_output_regressor.predict(X_val)

"""Loss Function => MSE"""

mse_val = mean_squared_error(y_val, y_val_pred)
print("Mean Squared Error (Validation):", mse_val)

y_test_pred = multi_output_regressor.predict(X_test)

mse_test = mean_squared_error(y_test, y_test_pred)
print("Mean Squared Error (Test):", mse_test)

#cv_scores = cross_val_score(multi_output_regressor, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
#mse_cv = -cv_scores.mean()
#print("Mean Squared Error (Cross-Validation):", mse_cv)

"""Loss Function => R^2"""

from sklearn.metrics import r2_score

r2_val = r2_score(y_val, y_val_pred)
print("R² Score (Validation):", r2_val)

r2_test = r2_score(y_test, y_test_pred)
print("R² Score (Test):", r2_test)

#cv_scores_r2 = cross_val_score(multi_output_regressor, X_train, y_train, scoring='r2', cv=5)
#r2_cv = cv_scores_r2.mean()
#print("R² Score (Cross-Validation):", r2_cv)

"""Alternative 2: One model per each KPI

Option 1. Lead time with RandomForestRegressor
"""

X = df_encoded.drop(['num__cancelation_ratio', 'num__parking_special_ratio', 'num__booking_changes_waiting_time_ratio',
               'num__adults_ratio', 'num__children_ratio', 'num__babies_ratio', 'num__special_requests_per_guest',
               'num__lead_time_previous_cancel_ratio', 'num__booking_changes_per_week', 'num__lead_time'], axis=1)

y = df_encoded['num__lead_time'];

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

random_forest_regressor_lead_time = RandomForestRegressor(random_state=42)

random_forest_regressor_lead_time.fit(X_train, y_train)

y_val_pred = random_forest_regressor_lead_time.predict(X_val)

"""Loss Function => MSE"""

mse_val = mean_squared_error(y_val, y_val_pred)
print("Mean Squared Error (Validation):", mse_val)

y_test_pred = random_forest_regressor_lead_time.predict(X_test)

mse_test = mean_squared_error(y_test, y_test_pred)
print("Mean Squared Error (Test):", mse_test)

#cv_scores = cross_val_score(random_forest_regressor_lead_time, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
#mse_cv = -cv_scores.mean()
#print("Mean Squared Error (Cross-Validation):", mse_cv)

"""Loss Function => R^2"""

from sklearn.metrics import r2_score

r2_val = r2_score(y_val, y_val_pred)
print("R² Score (Validation):", r2_val)

r2_test = r2_score(y_test, y_test_pred)
print("R² Score (Test):", r2_test)

#cv_scores_r2 = cross_val_score(random_forest_regressor_lead_time, X_train, y_train, scoring='r2', cv=5)
#r2_cv = cv_scores_r2.mean()
#print("R² Score (Cross-Validation):", r2_cv)

"""Option 2: Lead time with GradientBoostingRegressor"""

X = df_encoded.drop(['num__cancelation_ratio', 'num__parking_special_ratio', 'num__booking_changes_waiting_time_ratio',
               'num__adults_ratio', 'num__children_ratio', 'num__babies_ratio', 'num__special_requests_per_guest',
               'num__lead_time_previous_cancel_ratio', 'num__booking_changes_per_week', 'num__lead_time'], axis=1)

y = df_encoded['num__lead_time'];

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

from sklearn.ensemble import GradientBoostingRegressor

gb_regressor_lead_time = GradientBoostingRegressor()

gb_regressor_lead_time.fit(X_train, y_train)

y_val_pred_gb = gb_regressor_lead_time.predict(X_val)
y_test_pred_gb = gb_regressor_lead_time.predict(X_test)

"""Loss Function => R^2"""

r2_val_gb = r2_score(y_val, y_val_pred_gb)
print("R² Score (Validation - Gradient Boosting):", r2_val_gb)

r2_test_gb = r2_score(y_test, y_test_pred_gb)
print("R² Score (Test - Gradient Boosting):", r2_test_gb)

#cv_scores_r2_gb = cross_val_score(gb_regressor_lead_time, X_train, y_train, scoring='r2', cv=5)
#r2_cv_gb = cv_scores_r2_gb.mean()
#print("R² Score (Cross-Validation - Gradient Boosting):", r2_cv_gb)

"""Option 3: Lead Time with XGBoost"""

X = df_encoded.drop(['num__cancelation_ratio', 'num__parking_special_ratio', 'num__booking_changes_waiting_time_ratio',
               'num__adults_ratio', 'num__children_ratio', 'num__babies_ratio', 'num__special_requests_per_guest',
               'num__lead_time_previous_cancel_ratio', 'num__booking_changes_per_week', 'num__lead_time'], axis=1)

y = df_encoded['num__lead_time'];

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

num_rounds = 100
xg_boost_model = xgb.train(params, dtrain, num_rounds, evals=[(dval, 'validation')], early_stopping_rounds=10)

val_pred = xg_boost_model.predict(dval)

r2 = r2_score(y_val, val_pred)
print("R² Score (Test) using XGBoost:", r2)

predictions = xg_boost_model.predict(dtest)

r2 = r2_score(y_test, predictions)
print("R² Score (Test) using XGBoost:", r2)

"""# Critical Dimensions

Alternative 1: MultiOutputClassifier with RandomForestClassifier
"""

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

X = df_encoded.drop(['num__cancelation_ratio', 'num__parking_special_ratio', 'num__booking_changes_waiting_time_ratio',
               'num__adults_ratio', 'num__children_ratio', 'num__babies_ratio', 'num__special_requests_per_guest',
               'num__lead_time_previous_cancel_ratio', 'num__booking_changes_per_week', 'num__lead_time', 'cat__market_segment_Aviation',
               'cat__market_segment_Complementary', 'cat__market_segment_Corporate', 'cat__market_segment_Direct', 'cat__market_segment_Groups',
               'cat__market_segment_Offline TA/TO', 'cat__market_segment_Online TA', 'cat__market_segment_Undefined', 'cat__distribution_channel_Corporate',
               'cat__distribution_channel_Direct', 'cat__distribution_channel_GDS', 'cat__distribution_channel_TA/TO', 'cat__distribution_channel_Undefined'], axis=1)
y = df_encoded[['cat__market_segment_Aviation',
               'cat__market_segment_Complementary', 'cat__market_segment_Corporate', 'cat__market_segment_Direct', 'cat__market_segment_Groups',
               'cat__market_segment_Offline TA/TO', 'cat__market_segment_Online TA', 'cat__market_segment_Undefined', 'cat__distribution_channel_Corporate',
               'cat__distribution_channel_Direct', 'cat__distribution_channel_GDS', 'cat__distribution_channel_TA/TO', 'cat__distribution_channel_Undefined']]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

multi_output_classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, random_state=42))

multi_output_classifier.fit(X_train, y_train)

y_val_pred = multi_output_classifier.predict(X_val)

"""Loss Function => MSE"""

mse_val = mean_squared_error(y_val, y_val_pred)
print("Mean Squared Error (Validation):", mse_val)

y_test_pred = multi_output_classifier.predict(X_test)

mse_test = mean_squared_error(y_test, y_test_pred)
print("Mean Squared Error (Test):", mse_test)

#cv_scores = cross_val_score(multi_output_classifier, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
#mse_cv = -cv_scores.mean()
#print("Mean Squared Error (Cross-Validation):", mse_cv)

"""# Predictions

Data Preparation
"""

standard_scaler = pipeline.named_steps['preprocessor'].named_transformers_['num'].named_steps['scaler']

columns_to_drop = [
    'cat__market_segment_Aviation',
    'cat__market_segment_Complementary',
    'cat__market_segment_Corporate',
    'cat__market_segment_Direct',
    'cat__market_segment_Groups',
    'cat__market_segment_Offline TA/TO',
    'cat__market_segment_Online TA',
    'cat__market_segment_Undefined',
    'cat__distribution_channel_Corporate',
    'cat__distribution_channel_Direct',
    'cat__distribution_channel_GDS',
    'cat__distribution_channel_TA/TO',
    'cat__distribution_channel_Undefined'
]

new_df_encoded = df_encoded.drop(columns=columns_to_drop)
new_df_encoded.shape

"""Random Sample"""

inversed = standard_scaler.inverse_transform(new_df_encoded)
inversed_df = pd.DataFrame(data=inversed, columns=new_df_encoded.columns)

new_columns = {
    'cat__market_segment_Aviation': df_encoded['cat__market_segment_Aviation'],
    'cat__market_segment_Complementary': df_encoded['cat__market_segment_Complementary'],
    'cat__market_segment_Corporate': df_encoded['cat__market_segment_Corporate'],
    'cat__market_segment_Direct': df_encoded['cat__market_segment_Direct'],
    'cat__market_segment_Groups': df_encoded['cat__market_segment_Groups'],
    'cat__market_segment_Offline TA/TO': df_encoded['cat__market_segment_Offline TA/TO'],
    'cat__market_segment_Online TA': df_encoded['cat__market_segment_Online TA'],
    'cat__market_segment_Undefined': df_encoded['cat__market_segment_Undefined'],
    'cat__distribution_channel_Corporate': df_encoded['cat__distribution_channel_Corporate'],
    'cat__distribution_channel_Direct': df_encoded['cat__distribution_channel_Direct'],
    'cat__distribution_channel_GDS': df_encoded['cat__distribution_channel_GDS'],
    'cat__distribution_channel_TA/TO': df_encoded['cat__distribution_channel_TA/TO'],
    'cat__distribution_channel_Undefined': df_encoded['cat__distribution_channel_Undefined']
}

inversed_df = inversed_df.assign(**new_columns)

inversed_df.iloc[152]

"""KPIs' Prediction:"""

data = {
    'num__is_canceled': [0.000000],
    'num__lead_time': [0.000000],
    'num__arrival_date_year': [2015.000000],
    'num__arrival_date_week_number': [28.000000],
    'num__arrival_date_day_of_month': [5.000000],
    'num__stays_in_weekend_nights': [4.000000],
    'num__stays_in_week_nights': [6.000000],
    'num__adults': [3.000000],
    'num__children': [0.000000],
    'num__babies': [0.000000],
    'num__is_repeated_guest': [0.000000],
    'num__previous_cancellations': [0.000000],
    'num__previous_bookings_not_canceled': [0.000000],
    'num__booking_changes': [3.000000],
    'num__days_in_waiting_list': [0.000000],
    'num__adr': [124.450000],
    'num__required_car_parking_spaces': [1.000000],
    'num__total_of_special_requests': [1.000000],
    'num__cancelation_ratio': [0.0],
    'num__parking_special_ratio': [0.0],
    'num__booking_changes_waiting_time_ratio': [0.0],
    'num__adults_ratio': [0.0],
    'num__children_ratio': [0.0],
    'num__babies_ratio': [0.0],
    'num__special_requests_per_guest': [0.0],
    'num__lead_time_previous_cancel_ratio': [0.0],
    'num__booking_changes_per_week': [0.0],
}

df_for_sample = pd.DataFrame(data)

scaled_data = standard_scaler.transform(df_for_sample)
scaled_df = pd.DataFrame(scaled_data, columns=df_for_sample.columns)

scaled_df.iloc[0]

df_for_sample = df_for_sample.drop(['num__cancelation_ratio', 'num__parking_special_ratio', 'num__booking_changes_waiting_time_ratio',
               'num__adults_ratio', 'num__children_ratio', 'num__babies_ratio', 'num__special_requests_per_guest',
               'num__lead_time_previous_cancel_ratio', 'num__booking_changes_per_week', 'num__lead_time'], axis=1)
df_for_sample.info()

other_features = {
    'cat__market_segment_Aviation': [0.000000],
    'cat__market_segment_Complementary': [0.000000],
    'cat__market_segment_Corporate': [0.000000],
    'cat__market_segment_Direct': [1.000000],
    'cat__market_segment_Groups': [0.000000],
    'cat__market_segment_Offline TA/TO': [0.000000],
    'cat__market_segment_Online TA': [0.000000],
    'cat__market_segment_Undefined': [0.000000],
    'cat__distribution_channel_Corporate': [0.000000],
    'cat__distribution_channel_Direct': [1.000000],
    'cat__distribution_channel_GDS': [0.000000],
    'cat__distribution_channel_TA/TO': [0.000000],
    'cat__distribution_channel_Undefined': [0.000000]
}

other_features_df = pd.DataFrame(other_features)
scaled_df = pd.concat([scaled_df, other_features_df], axis=1)
scaled_df = scaled_df.drop(['num__cancelation_ratio', 'num__parking_special_ratio', 'num__booking_changes_waiting_time_ratio',
               'num__adults_ratio', 'num__children_ratio', 'num__babies_ratio', 'num__special_requests_per_guest',
               'num__lead_time_previous_cancel_ratio', 'num__booking_changes_per_week', 'num__lead_time'], axis=1)
scaled_df.iloc[0]

y_val_pred = multi_output_regressor.predict(scaled_df)
order = [
    'num__cancelation_ratio',
    'num__parking_special_ratio',
    'num__booking_changes_waiting_time_ratio',
    'num__adults_ratio',
    'num__children_ratio',
    'num__babies_ratio',
    'num__special_requests_per_guest',
    'num__lead_time_previous_cancel_ratio',
    'num__booking_changes_per_week',
    'num__lead_time'
]

df_kpis_regression = pd.DataFrame(data=y_val_pred, columns=order)
df_kpis_regression.iloc[0]

df_encoded.iloc[152][order]

"""Critical Dimensions' Prediction:"""

scaled_df.info()

new_scaled_df = scaled_df.drop(columns=columns_to_drop)
new_scaled_df.info()

y_val_pred = multi_output_classifier.predict(new_scaled_df)
cat_df = pd.DataFrame(data=y_val_pred, columns=columns_to_drop)
cat_df.iloc[0]

df_encoded.iloc[152][columns_to_drop]