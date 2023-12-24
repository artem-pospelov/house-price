import os

import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from torch.utils.tensorboard import SummaryWriter


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train_model(cfg: DictConfig):
    # Load data
    X_train = pd.read_csv(cfg.X_train_path)
    y_train = pd.read_csv(cfg.X_train_path)

    # Train model
    model = RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_split=cfg.min_samples_split,
        min_samples_leaf=cfg.min_samples_leaf,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)

    # Save model
    joblib.dump(model, cfg.model_path)

    mse = mean_squared_error(y_train, y_pred)
    mae = mean_absolute_error(y_train, y_pred)
    mape = mean_absolute_percentage_error(y_train, y_pred)

    y_train = pd.read_csv("y_train.csv")

    metrics_names = ["mse", "mae", "mape"]
    meitrics = [mse, mae, mape]

    for metric, name in zip(meitrics, metrics_names):
        sw.add_scalar(name, metric, global_step=0)

    return {"MSE": mse, "MAE": mae, "MAPE": mape}


if __name__ == "__main__":
    sw = SummaryWriter("exp_logs")
    os.system("dvc fetch X_train.csv")
    os.system("dvc fetch y_train.csv")
    os.system("dvc pull --remote myremote")
    train_model()
    os.system("dvc add model.joblib")
    os.system("dvc push model.joblib.dvc")

