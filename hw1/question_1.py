import pandas as pd
import numpy as np

# Read in data
train = pd.read_csv('train.csv')

# Create X and y, do the fitting calculation
X = train[['1stFlrSF','2ndFlrSF','TotalBsmtSF']].values
y = train[['SalePrice']].values
beta = ((np.linalg.inv(X.T.dot(X))).dot(X.T)).dot(y)
y_hat = X.dot(beta)

# Calculate R2 score
SS_res = np.sum((y-y_hat)**2)
SS_tot = np.sum((y-np.average(y))**2)
r2 = 1 - SS_res/SS_tot
print(r2)