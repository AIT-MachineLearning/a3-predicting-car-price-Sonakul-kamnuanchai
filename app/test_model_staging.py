import pytest
import numpy as np
import os
import mlflow
import cloudpickle as cp

#Load the model from mlflow
os.environ['MLFLOW_TRACKING_USERNAME'] = "admin"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "password"
mlflow.set_tracking_uri("https://mlflow.ml.brain.cs.ait.ac.th")
mlflow.set_experiment(experiment_name="st124738-a3")
model_name = "st124738-a3-model" 
model_version = '1'
loaded_model = mlflow.sklearn.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)

sample_input = np.array([[20, 15000, 2015, 50000, 1500, 110]])

def test_model_input():
    assert sample_input.shape == (1, 6), f"Expected input shape to be (1, 6) but got {sample_input.shape}"

def test_model_output_shape():
    intercept = np.ones((sample_input.shape[0], 1))
    sample_input_with_intercept = np.concatenate([intercept, sample_input], axis=1)
    prediction = loaded_model.predict(sample_input_with_intercept)
    assert prediction.shape == (1,), f"Expected output shape to be (1,) but got {prediction.shape}"

