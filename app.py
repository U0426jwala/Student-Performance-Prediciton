from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import logging
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")  # Replace with your MongoDB URI
db = client['flask_logs']  # Database name
collection = db['predictions']  # Collection name


def log_to_db(data):
    """
    Inserts logging data into MongoDB.
    :param data: Dictionary containing log details.
    """
    try:
        collection.insert_one(data)
    except Exception as e:
        logging.error(f"Failed to log data to MongoDB: {str(e)}")


@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Gather input data
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('writing_score')),
                writing_score=float(request.form.get('reading_score'))
            )

            pred_df = data.get_data_as_data_frame()
            logging.info("Input DataFrame created.")

            # Prediction pipeline
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            # Log successful prediction
            log_to_db({
                "status": "success",
                "input_data": pred_df.to_dict(orient='records'),
                "predictions": results.tolist(),
                "error": None
            })

            return render_template('home.html', results=results[0])

        except Exception as e:
            # Log error to MongoDB
            log_to_db({
                "status": "error",
                "input_data": None,
                "predictions": None,
                "error": str(e)
            })
            logging.error(f"Prediction error: {str(e)}")
            return render_template('home.html', results="An error occurred during prediction.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True, port=7860, host="0.0.0.0")