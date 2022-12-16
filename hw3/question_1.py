# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 22:16:18 2022

@author: ylwu5
"""

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Read in data
payments = pd.read_csv("data/payments.csv")
merchants = pd.read_csv("data/merchants.csv")
buyers = pd.read_csv("data/buyers.csv")


# ==== 1. Waiting For Chargebacks ====
print("PART 1")
# Select those with chargebacks
chargebacks = payments[~payments['chargeback_timestamp'].isnull()]

# Calculate using the known chargebacks for their time
time_until_chargeback = pd.to_datetime(chargebacks['chargeback_timestamp']) - pd.to_datetime(chargebacks['transaction_timestamp'])
sorted_time_until_chargeback = pd.DataFrame(np.sort(time_until_chargeback))

# Take time at the quantile 95% as the time for waiting
idx_of_95pct = int(sorted_time_until_chargeback.shape[0] * 0.95)
print(f'The index for the quantile 95% chargeback to happen is: {idx_of_95pct}.')
print("Check for date around this index:")
print(sorted_time_until_chargeback[0][idx_of_95pct])
print(sorted_time_until_chargeback[0][idx_of_95pct+1])
print("With rounding, the decision is to wait for 32 days for chargeback.")


# ==== 2. Train / Test Split ====
print("\nPART 2")
# Sort payments with transaction date, and reset dataframe index
sorted_payments = payments.sort_values(by=['transaction_timestamp'])
sorted_payments = sorted_payments.reset_index(drop=True)

# Find latest transaction date, and we want the latest 1 month's worth of data to be the test set
latest_date = pd.to_datetime(np.max(sorted_payments.transaction_timestamp))

# Get the start date for the test set
test_start_date = pd.to_datetime((latest_date - pd.Timedelta(30, 'd')).date())

# Also find the earliest date of transaction, as the start date of training set
training_start_date = pd.to_datetime(np.min(sorted_payments.transaction_timestamp))

# Print out results
print(f"The start timestamp of training set is: {training_start_date}; end timestamp is {test_start_date}.")
print(f"The start timestamp of test set is: {test_start_date}; end timestamp is {latest_date}.")


# ==== 3. Historical Features ====
print("\nPART 3")
# First add a column "is_fraud"
sorted_payments['is_fraud'] = sorted_payments['chargeback_timestamp'].notnull()

# Calculate fraud rate with window function
# Already ordered by transaction date
# fraud_rate = (current sum of frauds) / (current number of appearance of the merchant or buyer)
merchant_fraud_rate = (sorted_payments.groupby(['merchant_id']).cumsum()['is_fraud']) / (sorted_payments.groupby(['merchant_id']).cumcount() + 1)
buyer_fraud_rate = (sorted_payments.groupby(['buyer_id']).cumsum()['is_fraud']) / (sorted_payments.groupby(['buyer_id']).cumcount() + 1)

# Attach to the dataframe
rated_payments = sorted_payments.copy()
rated_payments['merchant_fraud_rate'] = merchant_fraud_rate
rated_payments['buyer_fraud_rate'] = buyer_fraud_rate

# To verify
sum_merchant_fraud_rate = np.sum(rated_payments['merchant_fraud_rate'])
sum_buyer_fraud_rate = np.sum(rated_payments['buyer_fraud_rate'])
print("To Verify:")
print(f"Sum of merchant fraud rate is: {sum_merchant_fraud_rate};")
print(f"Sum of buyer fraud rate is: {sum_buyer_fraud_rate};")
print(f"Sum of merchant plus sum of buyer fraud rate is: {sum_merchant_fraud_rate + sum_buyer_fraud_rate}.")


# ==== 4. Make a Model ====
print("\nPART 4")
# Change column names for better understanding when merging
merchants.rename(columns={"id":"merchant_id", "country":"merchant_country", "category":"merchant_category"}, inplace=True)
buyers.rename(columns={"id":"buyer_id", "country":"buyer_country"}, inplace=True)

# Merge payments, merchants, and buyers, get a dataframe that contains raw X and y
payments_merchants_temp = rated_payments.merge(merchants, on='merchant_id')
payments_merchants_buyers_temp = payments_merchants_temp.merge(buyers, on='buyer_id')

# Select useful part of data
# transaction_timestamp: to separate training and test set
# is_fraud: targeted y value
# others: features
raw_X_and_y = payments_merchants_buyers_temp[['transaction_timestamp', 'payment_amount', 'merchant_category', 'merchant_country', 'buyer_country', 'merchant_fraud_rate', 'buyer_fraud_rate', 'is_fraud']]

# Split training and test, using test_start_date from PART 2
raw_X_and_y_train = raw_X_and_y[pd.to_datetime(raw_X_and_y['transaction_timestamp']) < test_start_date]
raw_X_and_y_test = raw_X_and_y[pd.to_datetime(raw_X_and_y['transaction_timestamp']) >= test_start_date]

# Cut out X_train, y_train, X_test, y_test
raw_X_train = raw_X_and_y_train[['payment_amount', 'merchant_category', 'merchant_country', 'buyer_country', 'merchant_fraud_rate', 'buyer_fraud_rate']]
y_train = raw_X_and_y_train['is_fraud'] - 0
raw_X_test = raw_X_and_y_test[['payment_amount', 'merchant_category', 'merchant_country', 'buyer_country', 'merchant_fraud_rate', 'buyer_fraud_rate']]
y_test = raw_X_and_y_test['is_fraud'] - 0

# get all numeric features
ignore_features = []
features_num = [
    column
    for column, series in raw_X_train.items()
    if np.issubdtype(series.dtype, np.number) and column not in ignore_features
]
features_num
# get all categorical features
ignore_features_cat = [] + features_num      # ignore id, target, and all numerical features
features_cat = [column for column in raw_X_train.columns if column not in ignore_features_cat]
features_cat

# create pipeline
pipeline_num = Pipeline(
    [('imputer', SimpleImputer()),('scaler', StandardScaler())]
)
pipeline_cat = Pipeline(
    [('imputer', SimpleImputer(strategy='constant', fill_value='missing')),('onehot', OneHotEncoder(handle_unknown='ignore'))]
)
preprocessor = ColumnTransformer(
    [('num', pipeline_num, features_num),('cat', pipeline_cat, features_cat)]
)
pipeline = Pipeline(
    [('preprocessor', preprocessor), ('logistic',LogisticRegression(class_weight='balanced'))]
)

# Fit model
pipeline.fit(raw_X_train, y_train)
# Confusion Matrix
metrics.plot_confusion_matrix(pipeline, raw_X_test, y_test)
# ROC curve
metrics.plot_roc_curve(pipeline, raw_X_test, y_test)

