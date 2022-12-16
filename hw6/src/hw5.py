
from metaflow import FlowSpec, step, Parameter, IncludeFile, current
from datetime import datetime
import os
import numpy as np
import pandas as pd

# make sure we are running locally for this
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'


class ClassificationFlow(FlowSpec):
    '''
    For FRE-7773 homework 5, create a flow that trains a classification model,
    test the metrics, and implement fairness tests.
    '''

    DATA_FILE = IncludeFile(
        'dataset',
        help='Text file with the dataset',
        is_text=True,
        default='loan_dataset.csv'
    )

    TEST_SPLIT = Parameter(
        name='test_split',
        help='Determining the split of the dataset for testing',
        default = 0.3
    )

    RANDOM_STATE = Parameter(
        name='random_state',
        help='random state used in the program',
        default=42
    )

    @step
    def start(self):
        '''
        Start up and print out some info to make sure everything is ok metaflow-side
        '''
        print("Starting up at {}".format(datetime.utcnow()))
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)
        self.next(self.load_data)

    @step
    def load_data(self):
        '''
        Read the data in from the static file
        '''
        from io import StringIO
        self.df = pd.read_csv(StringIO(self.DATA_FILE))
        print(f'Shape of loaded data is {self.df.shape}')
        #self.y = self.df['Loan_Status'].values
        #self.y_trans = (self.y=="Y") + 0
        self.next(self.check_dataset)

    @step
    def check_dataset(self):
        '''
        Conditions can be checked here
        '''
        assert(all(y is not np.nan for y in self.df['Loan_Status'].values))
        self.next(self.separate_features)

    @step
    def separate_features(self):
        '''
        Separate numerical and categorical features
        
        '''
        features = self.df.columns.values
        self.ignore_features = ['Loan_ID','Loan_Status']
        self.num_features = []
        self.cat_features = []
        # categorical features
        for f in features:
            if f not in self.ignore_features and self.df[f].dtype == 'object':
                self.cat_features.append(f)
        # numerical features
        for f in features:
            if f not in self.ignore_features and f not in self.cat_features:
                self.num_features.append(f)
        print(self.num_features)
        print(self.cat_features)
        self.next(self.handle_missing)

    @step
    def handle_missing(self):
        '''
        Drop missing values since it is a small percentage of all the data
        OR
        Impute missing values differently for num and cat features
        ***Drop used in this homework***
        '''
        self.df = self.df.dropna()
        self.next(self.prepare_datasets_and_y)

    @step
    def prepare_datasets_and_y(self):
        self.df_num = self.df[self.num_features]
        self.df_cat = self.df[self.cat_features]
        self.y = self.df['Loan_Status'].values
        self.y_trans = (self.y == 'Y') + 0
        self.next(self.transform_combine)

    @step
    def transform_combine(self):
        '''
        Transform and combine num & cat features to build the X matrix
        '''
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import OneHotEncoder
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse=False)
        self.df_num_trans = self.scaler.fit_transform(self.df_num)
        self.df_cat_trans = self.encoder.fit_transform(self.df_cat)
        self.X = np.hstack((self.df_num_trans,self.df_cat_trans))
        self.next(self.prepare_train_and_test_dataset)

    @step
    def prepare_train_and_test_dataset(self):
        '''
        Train test split
        '''
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y_trans,
            test_size=self.TEST_SPLIT,
            random_state=self.RANDOM_STATE
        )
        self.next(self.train_log_model)

    @step
    def train_log_model(self):
        '''
        Train the logistic regression model, and print the f1 score
        '''
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import f1_score
        #self.log_model = LogisticRegression()
        #self.log_model = LogisticRegression(class_weight='balanced')
        self.log_model = RandomForestClassifier()
        self.log_model = self.log_model.fit(self.X_train, self.y_train)
        self.y_hat = self.log_model.predict(self.X_test)
        score = f1_score(self.y_test, self.y_hat)
        print(f'f1 score is {score}')
        self.next(self.fairness_test_features)

    @step
    def fairness_test_features(self):
        '''
        Calculate for all categorical features in this case.
        '''
        #self.fair_features = ['Gender', 'Property_Area']
        print(f'Doing fairness test for {self.cat_features}')
        self.next(self.fairness_test, foreach='cat_features')

    @step
    def fairness_test(self):
        '''
        Separate data into groups according to certain categorical feature,
        predict the number of approved loans and the rate of approval for each group,
        and generate a dictionary whose key is group and value is rate.
        '''
        self.test_feature = self.input
        group_col = self.df[self.test_feature]
        unique_groups = group_col.unique()

        X_groups = []
        group_names = []
        for group in unique_groups:
            group_names.append(group)
            X_groups.append(self.X[group_col==group])
        self.approve_rates = {}

        for i in range(len(group_names)):
            y_pred_approve = self.log_model.predict(X_groups[i])
            self.approve_rates[group_names[i]] = (y_pred_approve.sum() / len(X_groups[i]))

        self.next(self.join)
    
    @step
    def join(self, inputs):
        self.test_features = [input.test_feature for input in inputs]
        self.rates = [input.approve_rates for input in inputs]
        self.model = inputs[0].log_model
        self.scaler = inputs[0].scaler
        self.encoder = inputs[0].encoder
        self.df = inputs[0].df
        self.X_train = inputs[0].X_train
        self.y_train = inputs[0].y_train
        self.X_test = inputs[0].X_test
        self.y_test = inputs[0].y_test
        self.y_hat = inputs[0].y_hat
        

        self.fairness_result = {}
        for i in range(len(self.test_features)):
            self.fairness_result[self.test_features[i]] = self.rates[i]
        print(self.fairness_result)
        self.next(self.end)

    @step
    def end(self):
        # all done, just print goodbye
        print("All done at {}!".format(datetime.utcnow()))



if __name__ == '__main__':
    ClassificationFlow()