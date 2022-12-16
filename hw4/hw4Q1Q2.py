"""

Simple stand-alone script showing end-to-end training of a regression model using Metaflow. 
This script ports the composable script into an explicit dependency graphg (using Metaflow syntax)
and highlights the advantages of doing so. This script has been created for pedagogical purposes, 
and it does NOT necessarely reflect all best practices.

Please refer to the slides and our discussion for further context.

MAKE SURE TO RUN THIS WITH METAFLOW LOCAL FIRST

"""

# COMET_API_KEY=dKDuNGGljTjCbs9tsRILwY81C MY_PROJECT_NAME=yw3602_hw4_q1 python hw4Q1Q2.py run

from metaflow import FlowSpec, step, Parameter, IncludeFile, current
from datetime import datetime
from comet_ml import Experiment
import os

# MAKE SURE THESE VARIABLES HAVE BEEN SET
assert 'COMET_API_KEY' in os.environ and os.environ['COMET_API_KEY']
assert 'MY_PROJECT_NAME' in os.environ and os.environ['MY_PROJECT_NAME']
print("Running experiment for project: {}".format(os.environ['MY_PROJECT_NAME']))

# make sure we are running locally for this
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'


class RidgeFlow(FlowSpec):
    """
    Modified from SampleRegressionFlow, using Ridge instead.
    SampleRegressionFlow is a minimal DAG showcasing reading data from a file 
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
        default='regression_dataset.txt')

    TEST_SPLIT = Parameter(
        name='test_split',
        help='Determining the split of the dataset for testing',
        default=0.20
    )

    VALIDATION_SPLIT = Parameter(
        name='validation_split',
        help='Determining the split of the dataset for validation from the trainning set',
        default=0.20
    )

    # alphas = [0.01, 0.1, 1, 10]
    ALPHA_PARAMS = Parameter(
        name='alpha_params',
        help='Alpha values for Ridge hyperparameter, in string type separated by comma',
        default="0.01,0.1,1,10"
    )


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
        self.rand_state = 42
        self.next(self.load_data)

    @step
    def load_data(self): 
        """
        Read the data in from the static file
        """
        from io import StringIO

        raw_data = StringIO(self.DATA_FILE).readlines()
        print("Total of {} rows in the dataset!".format(len(raw_data)))
        self.dataset = [[float(_) for _ in d.strip().split('\t')] for d in raw_data]
        print("Raw data: {}, cleaned data: {}".format(raw_data[0].strip(), self.dataset[0]))
        self.Xs = [[_[0]] for _ in self.dataset]
        self.Ys =  [_[1] for _ in self.dataset]
        # go to the next step
        self.next(self.check_dataset)

    @step
    def check_dataset(self):
        """
        Check data is ok before training starts
        """
        assert(all(y < 100 and y > -100 for y in self.Ys))
        self.next(self.prepare_train_and_test_dataset)

    @step
    def prepare_train_and_test_dataset(self):
        from sklearn.model_selection import train_test_split

        # Divide train+validation and test
        self.X_tv, self.X_test, self.y_tv, self.y_test = train_test_split(
            self.Xs,
            self.Ys,
            test_size=self.TEST_SPLIT,
            random_state=self.rand_state
            )

        # Divide train and validation
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_tv,
            self.y_tv,
            test_size=self.VALIDATION_SPLIT,
            random_state=self.rand_state
        )
        
        #convert input hyperparameters to list
        self.alphas = [float(value) for value in self.ALPHA_PARAMS.split(',')]
        self.next(self.train_validation_model, foreach="alphas")

    @step
    def train_validation_model(self):
        """
        Train a regression on the training set
        """
        self.alpha = self.input

        from sklearn import linear_model
        reg = linear_model.Ridge(alpha=self.alpha)

        reg.fit(self.X_train, self.y_train)
        print("Coefficient {}, intercept {}".format(reg.coef_[0], reg.intercept_))
        # now, make sure the model is available downstream
        self.model = reg
        # go to the testing phase
        self.next(self.validate_model)

    @step
    def validate_model(self):
        '''
        Calculate metrics for the run with a certain hyperparameter
        '''
        from sklearn import metrics
        self.y_val_hat = self.model.predict(self.X_val)
        self.r2_val = metrics.r2_score(self.y_val, self.y_val_hat)
        self.mse_val = metrics.mean_squared_error(self.y_val, self.y_val_hat)
        self.mae_val = metrics.mean_absolute_error(self.y_val, self.y_val_hat)
        exp_val = Experiment(
            api_key=os.environ['COMET_API_KEY'],
            project_name=os.environ['MY_PROJECT_NAME'],
            auto_param_logging=False
        )
        self.metrics_val = {
            "Alpha": self.alpha,
            "R2": self.r2_val,
            "MSE": self.mse_val,
            "MAE": self.mae_val
        }
        exp_val.log_metrics(self.metrics_val)
        exp_val.log_parameter("Alpha", self.alpha)
        exp_val.set_name(f"Val Alpha = {self.alpha}")
        # Go to join the validation results and choose the best hyperparameter
        self.next(self.join)

    @step
    def join(self, inputs):
        '''
        Join models from validation step,
        Choose the best parameter according to r2 score
        '''
        self.val_alphas = [input.alpha for input in inputs]
        self.val_r2s = [input.r2_val for input in inputs]
        self.val_mse = [input.mse_val for input in inputs]
        self.val_mae = [input.mae_val for input in inputs]

        # Also need to keep the dataset split
        self.X_tv = inputs[0].X_tv
        self.X_test = inputs[0].X_test
        self.y_tv = inputs[0].y_tv
        self.y_test = inputs[0].y_test

        import numpy as np
        best_param_idx = np.argmax(self.val_r2s)
        self.best_alpha = self.val_alphas[best_param_idx]
        # Go to next step and train a model with the best parameter
        self.next(self.train_best_model)
        
    @step
    def train_best_model(self):
        '''
        Train a model with the best parameter chosen from previous step
        '''

        from sklearn import linear_model
        reg = linear_model.Ridge(alpha=self.best_alpha)
        reg.fit(self.X_tv, self.y_tv)
        print("Coefficient {}, intercept {}".format(reg.coef_[0], reg.intercept_))
        self.best_model = reg
        self.next(self.test_model)

    @step 
    def test_model(self):
        """
        Test the model on the hold out sample
        """
        from sklearn import metrics

        self.y_predicted = self.best_model.predict(self.X_test)
        self.r2 = metrics.r2_score(self.y_test, self.y_predicted)
        self.mse = metrics.mean_squared_error(self.y_test, self.y_predicted)
        self.mae = metrics.mean_absolute_error(self.y_test, self.y_predicted)
        print('MSE is {}, MAE is {}, R2 score is {}'.format(self.mse, self.mae, self.r2))
        # print out a test prediction
        test_predictions = self.best_model.predict([[10]])
        print("Test prediction is {}".format(test_predictions))
        # all is done go to the end
        self.next(self.report)

    @step
    def report(self):
        '''
        Create an Experiment object, and upload a report to comet
        '''
        exp = Experiment(api_key=os.environ['COMET_API_KEY'],
                        project_name=os.environ['MY_PROJECT_NAME'],
                        auto_param_logging=False)

        params={
            "Model": "Ridge",
            "Best Alpha": self.best_alpha
        }
        metrics={
            "mse": self.mse,
            "mae": self.mae,
            "r2": self.r2,
        }
        exp.set_name("Best Model")
        exp.log_dataset_hash(self.X_tv)
        exp.log_parameters(params)
        exp.log_metrics(metrics)

        self.next(self.end)


    @step
    def end(self):
        # all done, just print goodbye
        print("All done at {}!\n See you, space cowboys!".format(datetime.utcnow()))


if __name__ == '__main__':
    RidgeFlow()
