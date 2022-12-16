from flask import (
    Flask,
    render_template,
    request,
    send_file,
    send_from_directory,
    jsonify,
)

from dev import DoNothing
import joblib
import numpy as np
import pandas as pd
from metaflow import Flow
from metaflow import get_metadata, metadata
import json

FLOW_NAME = 'MyRegressionFlow' # name of the target class that generated the model
# Set the metadata provider as the src folder in the project,
# which should contains /.metaflow
metadata('./')
# Fetch currently configured metadata provider to check it's local!
print(get_metadata())

def get_latest_successful_run(flow_name: str):
    "Gets the latest successfull run."
    for r in Flow(flow_name).runs():
        if r.successful: 
            return r

# get artifacts from latest run, using Metaflow Client API
latest_run = get_latest_successful_run(FLOW_NAME)
model = latest_run.data.final_model

# We need to initialise the Flask object to run the flask app 
# By assigning parameters as static folder name,templates folder name
app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route("/", methods = ["GET"])
def main():
    if request.method == "GET":
        return render_template("index.html")

@app.route("/predict", methods = ["POST", "GET"])
def predict():
    resp = json.loads(request.data)["features"]
    if resp['gender'] == 'Male':
        resp['gender'] = 'M'
    else:
        resp['gender'] = 'F'
    resp["frozen_sqft"] = float(resp["frozen_sqft"])
    resp["meat_sqft"] = float(resp["meat_sqft"])
    items = ['coffee_bar', 'video_store', 'salad_bar', 'prepared_food', 'florist']
    for item in items:
        resp[item] = resp[item].lower()
        if resp[item] == 'yes':
            resp[item] = float('1')
        else:
            resp[item] = float('0')

    data = pd.Series(resp).to_frame().T
    pred = model.predict(data).tolist()
    print(pred)
    return str(pred[0])

@app.route('/json_predict',methods=['GET'])
def json_predict():
    # passing arguments with URL
    promotion_name = request.args.get('promotion_name', default = "Bag Stuffers", type=str)
    media_type = request.args.get('media_type', default="Bulk Mail", type=str)
    gender = request.args.get('gender', default="F", type=str)
    # process inputs
    promotion_name = " ".join(promotion_name.split("_"))
    media_type = " ".join(media_type.split("_"))

    # For manual input simplicity in this project, we only take 3 variables from the URL.
    # If done by machine input, this can be changed to use all the varibales.
    input = {
        "promotion_name": promotion_name,
        "media_type": media_type,
        "gender": gender,
        "frozen_sqft": 5312.9,  # took mean of all the data
        "meat_sqft": 3541.8,    # took mean of all the data
        "coffee_bar": 1.0,
        "video_store": 1.0,
        "salad_bar": 1.0,
        "prepared_food": 1.0,
        "florist": 1.0
    }
    s_input = pd.Series(input).to_frame().T

    if request.method=='GET':
        val = model.predict(s_input)

    # Example: URL/json_predict?promotion_name=One_Day_Sale&media_type=Daily_Paper,_Radio,_TV&gender=M
    # Returning the response to the client	
    resp = {
      "Cost": val[0],
    }
    return jsonify(resp)


if __name__ == "__main__":
    app.run(debug=True)