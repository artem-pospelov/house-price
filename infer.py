import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.tensorboard import SummaryWriter

def run_inference(summary_writer):
    # Load model
    model = joblib.load('model.joblib')

    # Load input data
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.to_csv('predicted.csv', index=False)
    
    # Save metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    metrics_names = ["mse", "mae", "rmse"]
    meitrics = [mse, mae, rmse]

    for metric, name in zip(meitrics, metrics_names):
        summary_writer.add_scalar(name, metric, global_step=0)

    return y_pred


if __name__ == "__main__":
    summary_writer = SummaryWriter("exp_logs")
    os.system("dvc fetch model.joblib")
    os.system("dvc pull --remote myremote")
    run_inference(summary_writer)
    os.system("dvc add predicted.csv")
    os.system("dvc push predicted.csv.dvc")
