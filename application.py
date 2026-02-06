import pickle
from flask import Flask,render_template,request
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),  # FIXED
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score', 0)),
            writing_score=float(request.form.get('writing_score', 0)),
        )

        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        final_result = float(results[0])  # SAFE
        return render_template('home.html', results=final_result)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)