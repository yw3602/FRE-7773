Team Project by Yilun Wu (yw3602), Zepeng Liu (zl4306), Bin Li (bl3384).

Dataset from Kaggle: https://www.kaggle.com/datasets/ramjasmaurya/medias-cost-prediction-in-foodmart
Each row of data corresponds to one certain customer, store, promotion and product.

To run the project:
1. Install environment according to requirements.txt.
2. Run dev.py from terminal using "python dev.py run", to run a metaflow flow and generate the metaflow folder which includes the runs.
    "dev_with_comet.py" adds connection to Comet ML, and requires a COMET_API_KEY and MY_PROJECT_NAME when running in terminal.
    Example: COMET_API_KEY=yourapikey MY_PROJECT_NAME=yourprojectname python dev_with_comet.py run.
3. Run app.py to start the python server.
4. Go to the website indicated in the terminal as the server starts. Input and predict.

To obtain a json response, use URL/json_predict and pass the variables promotion_name, media_type, and gender. Use "F" or "M" to indicate "Female" or "Male".
Example: URL/json_predict?promotion_name=One_Day_Sale&media_type=Daily_Paper,_Radio,_TV&gender=M