import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn import metrics

# read in data
initial_file_data = pd.read_csv("../data/airline_satisfaction/train.csv", index_col=0)

# get all numeric features
target = "satisfaction"
ignore_features = ["id","satisfaction"]
features_num = [
    column
    for column, series in initial_file_data.items()
    if np.issubdtype(series.dtype, np.number) and column not in ignore_features
]
# get all categorical features
ignore_features_cat = ["id","satisfaction"] + features_num      # ignore id, target, and all numerical features
features_cat = [column for column in initial_file_data.columns if column not in ignore_features_cat]

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
    [('preprocessor', preprocessor), ('logistic',LogisticRegression())]
)

# get X and y
features = [col for col in initial_file_data.columns if col not in ignore_features]
X_raw = initial_file_data[features]
y = LabelEncoder().fit_transform(initial_file_data[target])
# split
X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.3, random_state=42)

# gridsearch
c_vals = np.logspace(-1, 6, 8)
param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'logistic__C': c_vals,
    'logistic__fit_intercept': [True, False]
}
grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, return_train_score=True
)
grid_search = grid_search.fit(X_train, y_train)

#metrics.plot_roc_curve(grid_search, X_test, y_test)
optimal_params = grid_search.best_params_
print('The best parameters according to grid search are:')
print(optimal_params)

# ==== After grid search finds the best params ====
# create an optimal pipeline with the optimal parameters from grid search
pipeline_num = Pipeline(
    [('imputer', SimpleImputer(strategy=optimal_params['preprocessor__num__imputer__strategy'])),('scaler', StandardScaler())]
)
pipeline_cat = Pipeline(
    [('imputer', SimpleImputer(strategy='constant', fill_value='missing')),('onehot', OneHotEncoder(handle_unknown='ignore'))]
)
preprocessor = ColumnTransformer(
    [('num', pipeline_num, features_num),('cat', pipeline_cat, features_cat)]
)
pipeline = Pipeline(
    [('preprocessor', preprocessor), ('logistic',LogisticRegression(C=optimal_params['logistic__C'],fit_intercept=optimal_params['logistic__fit_intercept']))]
)

# train a model on the full training dataset with optimal hyperparameters
pipeline.fit(X_train,y_train)

# predict on test set (not used in drawing metrics function of drawing roc)
y_pred = pipeline.predict(X_test)

# draw roc curve for test set
metrics.plot_roc_curve(pipeline, X_test, y_test)

