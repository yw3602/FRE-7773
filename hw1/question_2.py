import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

import matplotlib.pyplot as plt

# Features to use and corresponding y
train = pd.read_csv('train.csv')
features = train[[
    "1stFlrSF",
    "2ndFlrSF",
    "TotalBsmtSF",
    "LotArea",
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
]].values
y = train[['SalePrice']].values

# Train 8 models with increasing number of features
models = []
y_hats = []
for i in range(1,9):
    lnrModel = LinearRegression(fit_intercept=True)
    X = features[:,:i]
    #print(X.shape)
    lnrModel = lnrModel.fit(X,y)
    models.append(lnrModel)
    y_hats.append(lnrModel.predict(X))

# 4 Required metrics
r2s = []
MSEs = []
MAEs = []
MAPEs = []
for y_hat in y_hats:
    r2s.append(r2_score(y,y_hat))
    MSEs.append(mean_squared_error(y,y_hat))
    MAEs.append(mean_absolute_error(y,y_hat))
    MAPEs.append(mean_absolute_percentage_error(y,y_hat))

# Plotting settings
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 18

# Plotting function
def plot_metrics_features(metrics,metrics_name):
    plt.scatter(np.arange(1,9), metrics)
    plt.ylabel(metrics_name)
    plt.xlabel("Number of Features")
    return plt

# Draw plots
metrics_names = ['R2 score','Mean Squared Error','Mean Absolute Error','Mean Absolute Percentage Error']
metrics = [r2s,MSEs,MAEs,MAPEs]
for i in range(4):
    plot_metrics_features(metrics[i],metrics_names[i])
    plt.show()