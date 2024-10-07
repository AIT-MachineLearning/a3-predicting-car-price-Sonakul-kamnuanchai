#!/bin/bash

# Print each command before executing it (for debugging)
set -x

# Start the MLflow server
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root /mlflow/artifacts \
    --host 0.0.0.0 \
    --port 5000
