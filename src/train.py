import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import sys

with __name__ == "__main__":
    input_file = sys.argv[1]
    model_file = sys.argv[2]
    params_file = sys.argv[3]

    train(input_file, model_file, params_file)
