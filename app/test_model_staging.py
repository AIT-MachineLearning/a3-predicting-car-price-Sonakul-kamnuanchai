import pytest
import numpy as np
from app import loaded_model

sample_input = np.array([[20, 15000, 2015, 50000, 1500, 110]])

def test_model_input():
    assert sample_input.shape == (1, 6), f"Expected input shape to be (1, 6) but got {sample_input.shape}"

def test_model_output_shape():
    intercept = np.ones((sample_input.shape[0], 1))
    sample_input_with_intercept = np.concatenate([intercept, sample_input], axis=1)
    prediction = loaded_model.predict(sample_input_with_intercept)
    assert prediction.shape == (1,), f"Expected output shape to be (1,) but got {prediction.shape}"

