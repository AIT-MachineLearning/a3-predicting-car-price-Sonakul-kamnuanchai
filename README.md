# a3-predicting-car-prices-Sonakul-kamnuanchai
Sonakul kamnuanchai st124738

# Objectives :
  - Modify Multinomial Logistic Regression(LogisticRegression, Ridge-LogisticRegression) model and using method (stochastic, mini-batch, batch).
  - Using classification report (accuracy,precision,recall,f1-score,macro-precision,macro-recall,macro-f1-score,weighted-precision,weighted-recall,weighted-f1-score) to compare our custom classification report and classification report in sklearn.
  - Creating automated test file before deployment
  - Logging model,register (make sure register model on staging) and select the best model from mlflow in server
  - Using CI/CD for automated test and deployment

# Files locate:
 - Datasets : app/datasets
 - Jupyter Notebook : code/a3-redicting-car-prices-sonakul-kamnuanchai/A03_Carprediction_Sonakul_st124738.ipynb
 - automated test file : app/test_model_staging.py
 - Model : app/model/GGCars3.supermodel
 - Webapp : app/app.py
 - Dockerfiles : app/Dockerfile

# There are params will compared to find the best regression type in Multinomial Logistic Regression model:
  - regs : "LogisticRegression","Ridge-LogisticRegression"
  - batch_methods : stochastic, mini-batch, batch

# How to run automated test file in your local
  - pytest test_model_staging.py

# How to run in your local machine:
Go to Folder app and open terminal
  - command : docker build -t your-image-name . (Build your images)
  - command : docker compose up (run website in your local machine)

 Website a3 car prediction : https://st124738-a3.ml.brain.cs.ait.ac.th

