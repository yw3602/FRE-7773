"""

Simple stand-alone script showing end-to-end training of a regression model using Metaflow. 
This script ports the composable script into an explicit dependency graphg (using Metaflow syntax)
and highlights the advantages of doing so. This script has been created for pedagogical purposes, 
and it does NOT necessarely reflect all best practices.

Please refer to the slides and our discussion for further context.

MAKE SURE TO RUN THIS WITH METAFLOW LOCAL FIRST

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from metaflow import FlowSpec, step, Parameter, IncludeFile, current
from datetime import datetime
import os


# make sure we are running locally for this
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'


class DoNothing:
    def __init__(self):
        pass
    def fit(self, df, y=None):
        return self
    def transform(self, df):
        return df

def num_features_proc(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num_df = df.select_dtypes(include=numerics)
    num_features = list(num_df.columns)
    return num_features

def binary_num_features_proc(num_features, df):
    binary_num_features = []
    for feature in num_features:
        if (len(df[feature].unique()) == 2):
            binary_num_features.append(feature)
    return binary_num_features

def cat_featrues_proc(df):
    cat_df = df.select_dtypes(include="object")
    cat_features = list(cat_df.columns)
    print(cat_df.shape)
    return cat_features

def feature_preprocess(df):
    num_features = num_features_proc(df)
    binary_num_features = binary_num_features_proc(num_features, df)
    num_features = [f for f in num_features if f not in binary_num_features]
    cat_featrues = cat_featrues_proc(df)

    no_changer = DoNothing()

    pipeline_num = Pipeline(
        [('Scaler', StandardScaler())]
    )
    pipeline_bi = Pipeline(
        [('Do Nothing', no_changer)]
    )
    pipeline_cat = Pipeline(
        [("One-Hot Encoder", OneHotEncoder())]
    )

    # deal with three types of feature using different preprocessor, then combine
    preprocessor = ColumnTransformer([
        ('num', pipeline_num, num_features),
        ('binary_num', pipeline_bi, binary_num_features),
        ('cat', pipeline_cat, cat_featrues)
    ])
    print(f"features are {num_features + binary_num_features + cat_featrues}")
    return preprocessor

class MyRegressionFlow(FlowSpec):
    """
    MyRegressionFlow is a minimal DAG showcasing reading data from a file 
    and training a model successfully.
    """
    
    # if a static file is part of the flow, 
    # it can be called in any downstream process,
    # gets versioned etc.
    # https://docs.metaflow.org/metaflow/data#data-in-local-files
    DATA_FILE = IncludeFile(
        'dataset',
        help='Text file with the dataset',
        is_text=True,
        default='media prediction and its cost.csv')

    TEST_SPLIT = Parameter(
        name='test_split',
        help='Determining the split of the dataset for testing',
        default=0.20
    )

    # Test and compare three models.
    model_linear = LinearRegression()
    model_svm = SVR()
    model_forest = RandomForestRegressor()
    model_tuples = [("Linear Regression",model_linear),
                    ("Support Vector Mechine",model_svm),
                    ("Random Forest",model_forest)]

    @step
    def start(self):
        """
        Start up and print out some info to make sure everything is ok metaflow-side
        """
        print("Starting up at {}".format(datetime.utcnow()))
        # debug printing - this is from https://docs.metaflow.org/metaflow/tagging
        # to show how information about the current run can be accessed programmatically
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)
        self.next(self.load_data)

    @step
    def load_data(self): 
        """
        Read the data in from the static file
        """
        # pylint: disable=no-member
        
        # droping null values in the beginning, and we are not using any inputer in this program
        self.full_df = pd.read_csv('media prediction and its cost.csv').dropna()
        #Get y
        self.y = self.full_df['cost'].values

        # Using the selected features from "Feature Engineering.ipynb".
        self.df = self.full_df[["promotion_name","media_type","gender",
                "frozen_sqft","meat_sqft","coffee_bar","video_store","salad_bar","prepared_food","florist"]]
        
        # go to the next step
        self.next(self.prepare_dataset)
        
    @step
    def prepare_dataset(self):
        '''
        Using pipeline, so we can just take the raw data as X.
        '''
        self.X = self.df
        # go to the next step
        self.next(self.split_dataset)
        

    @step
    def split_dataset(self):
        '''
        We are testing and comparing differet models, so we have train test split and validation split.
        '''
        VALIDATION_SIZE = 0.3
        TEST_SIZE = 0.2
        RANDOM_STATE = 42
        
        # split train+val and test
        self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(self.X,self.y,
                                                                                        test_size=TEST_SIZE,random_state=RANDOM_STATE)

        # split train and val
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train_val,self.y_train_val,
                                                                              test_size=VALIDATION_SIZE,random_state=RANDOM_STATE)
        self.feature_preprocessor = feature_preprocess(self.X)
        self.next(self.train_model, foreach='model_tuples')
        #self.next(self.train_linear, self.train_DT, self.train_SVM, self.train_RF)

    @step 
    def train_model(self):
        '''
        For each branch, train the corresponding model
        '''
        self.model_input = self.input
        self.model_name = self.model_input[0]
    
        print(f"fitting {self.model_name}")

        self.model = make_pipeline(
            self.feature_preprocessor,
            self.model_input[1]
        )
        self.model.fit(self.X_train,self.y_train)
        self.next(self.valid_model)

    @step
    def valid_model(self):
        '''
        For each branch, validate the model, get the r2 score and other metrics
        '''
        self.y_val_pred = self.model.predict(self.X_val)
        self.model_r2 = r2_score(self.y_val,self.y_val_pred)
        self.model_MSE = mean_squared_error(self.y_val, self.y_val_pred)
        
        print(f"{self.model_name} r2 is: ", self.model_r2)
        print(f"{self.model_name} MSE is: ", self.model_MSE)
        self.res={"model":self.model_input,"r2":self.model_r2,"MSE":self.model_MSE}
        self.next(self.join)
        
    @step
    def join(self,inputs):
        '''
        Join the branches, choose the model with the highest r2 score as the best model
        '''
        self.r=[input.res for input in inputs]              # Model results. A list of dict
        self.mses=[input.model_MSE for input in inputs]     # Model MSE values
        self.r2s=[input.model_r2 for input in inputs]       # Model r2 scores

        self.best_r2= float("-inf")
        self.best_model = None

        for res in self.r:
            if self.best_r2 < res["r2"]:
                self.best_r2 = res["r2"]
                self.best_model = res["model"]
        print("Best model is {}".format(self.best_model[0]))
        self.merge_artifacts(inputs, include=['X','y'])
        self.next(self.train_best_model)

    @step 
    def train_best_model(self):
        """
        Train a final model (with pipeline) using the chosen best model.
        This model is used in the app.
        """
        VALIDATION_SIZE = 0.3
        TEST_SIZE = 0.2
        RANDOM_STATE = 42
        self.df_train, self.df_test, self.df_y_train, self.df_y_test = train_test_split(self.X,self.y,
                                                                                        test_size=TEST_SIZE,random_state=RANDOM_STATE)
        self.feature_preprocessor = feature_preprocess(self.X)
        self.final_model = make_pipeline(
            self.feature_preprocessor,
            self.best_model[1]
        )
        self.final_model = self.final_model.fit(self.df_train, self.df_y_train)
        y_test_pred = self.final_model.predict(self.df_test)
        self.best_r2 = r2_score(self.df_y_test, y_test_pred)
        print("best r2 is: ", self.best_r2)
        # all is done go to the end
        self.next(self.fairness_test)
    
    @step
    def fairness_test(self):
        '''
        Test the fairness of our model on gender.
        We use two dataframes who have the same data but have different value in "gender"
        One only Male and one only Female
        Then we compare the average of the predicted cost, to see if the difference is acceptable.
        '''
        # Redo import the data, as in load_data
        self.full_df = pd.read_csv('media prediction and its cost.csv').dropna()
        self.df = self.full_df[["promotion_name","media_type","gender",
                "frozen_sqft","meat_sqft","coffee_bar","video_store","salad_bar","prepared_food","florist"]]
        # create male and female dataframe
        self.df_male = self.df.copy()
        self.df_male['gender'] = 'M'
        self.df_female = self.df.copy()
        self.df_female['gender'] = 'F'

        y_pred_male = self.final_model.predict(self.df_male)
        y_mean_male = y_pred_male.mean()
        y_pred_female = self.final_model.predict(self.df_female)
        y_mean_female = y_pred_female.mean()

        fairness_result = {
            "Male": y_mean_male,
            "Female": y_mean_female
        }
        print("Fairness test:")
        print(fairness_result)

        self.next(self.end)


    @step
    def end(self):
        # all done, just print goodbye
        print("All done at {}!\n See you, space cowboys!".format(datetime.utcnow()))


if __name__ == '__main__':
    MyRegressionFlow()

