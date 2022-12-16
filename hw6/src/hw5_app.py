"""

This a streamlit app that retrieves the latest Metaflow local run, displays data rows
and make available a simple interactive UI for people to test the model!

Documenting your model, through cards, and make its predictions easily accessible
is a fundamental component of building trust in your model across your organization! 

"""

# import libraries
import streamlit as st
from metaflow import Flow
from metaflow import get_metadata, metadata
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# make sure we point the app to the flow folder
# NOTE: MAKE SURE TO RUN THE FLOW AT LEAST ONE BEFORE THIS ;-)
FLOW_NAME = 'ClassificationFlow' # name of the target class
# Set the metadata provider as the src folder in the project,
# which should contains /.metaflow
metadata('./')
# Fetch currently configured metadata provider - check it's local!
print(get_metadata())

# build up the dashboard
st.markdown("# ClassificationFlow playground")
st.write("This application shows the dataset and predictions made by our model!")

@st.cache
def get_latest_successful_run(flow_name: str):
    "Gets the latest successfull run."
    for r in Flow(flow_name).runs():
        if r.successful: 
            return r

# get artifacts from latest run, using Metaflow Client API
latest_run = get_latest_successful_run(FLOW_NAME)
latest_model = latest_run.data.model
latest_df = latest_run.data.df
latest_X_test = latest_run.data.X_test
latest_y_test = latest_run.data.y_test
latest_y_hat = latest_run.data.y_hat
latest_fairness = latest_run.data.fairness_result
latest_scaler = latest_run.data.scaler
latest_encoder = latest_run.data.encoder
#latest_X_train = latest_run.data.X_train
#y_predicted = latest_run.data.y_predicted
#y_test = latest_run.data.y_test

# show dataset
st.markdown("## Dataset")
st.write("First 3 Xs (before transformation) in the dataset:")
st.write(latest_df.head(3))

# model score
st.markdown("## Model Score")
st.write("Model Chosen: Random Forest Classification")
f1 = f1_score(latest_y_test, latest_y_hat)
st.write(f"F1 score: {f1}")

# confusion matrix
st.markdown("## Confusion Matrix")
st.write("Confusion matrix for the result of the test set")
plot_confusion_matrix(latest_model, latest_X_test, latest_y_test)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

# show fairness test result
st.markdown("## Fairness Test")
st.write("We divided the data into groups according to certain categorical feature, do a prediction on each group, and calculate their rate of getting approved of the loan (ie. getting an 1 as prediction result).\nThe result of the fairness test is below:")
option = st.selectbox("Category", ('Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'))
group_performance = pd.Series(latest_fairness[option])
st.write(group_performance)
plt.bar(group_performance.index, height=group_performance)
plt.title(option)
plt.ylim(0,1)
st.pyplot()

# play with the model
#st.markdown("## Model")
#_x = st.text_input('Input value (float):', '10.0')
#val = latest_model.predict([[float(_x)]])
#st.write('Input is {}, prediction is {}'.format(_x, val))