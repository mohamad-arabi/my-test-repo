# Databricks notebook source
import numpy as np
from sklearn.linear_model import LinearRegression

import mlflow
# enable autologging
mlflow.set_experiment(experiment_id='39a89da4f33143d4bff6b9d649327166')
mlflow.sklearn.autolog()

# prepare training data
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3
# train a model
model = LinearRegression()
with mlflow.start_run() as run:
    model.fit(X, y)
    mlflow.log_params({"param key": "param val 2"})
    mlflow.log_metrics({"metric name": 2})
    mlflow.set_tags({"tag key": "tag val 2"})
