import os

import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def open_data(path, target):
    df = pd.read_csv(path)
    features = df.drop(columns=[target], axis=1)
    target = df[target]

    return features, target

def split_data(features, target, test_size, random_state, X_save, y_save):
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
        )
    X_test_df = pd.DataFrame(X_test)
    X_test_df.to_csv(X_save, index=False)
    y_test_df = pd.DataFrame(y_test)
    y_test_df.to_csv(y_save, index=False)
    
    return X_train, X_test, y_train, y_test

@hydra.main(version_base=None, config_path="configs", config_name="config")
def train_model(cfg: DictConfig):
    # Load data
    features, target = open_data(cfg.paths.data, cfg.preparation.y_column)
    X_train, X_test, y_train, y_test = split_data(
        features, target, 
        cfg.preparation.test_size, 
        cfg.preparation.random_state,
        cfg.paths.X_test,
        cfg.paths.y_test
        )

    # Train model
    model = RandomForestRegressor(
        n_estimators=cfg.model_parameters.n_estimators,
        max_depth=cfg.model_parameters.max_depth,
        min_samples_split=cfg.model_parameters.min_samples_split,
        min_samples_leaf=cfg.model_parameters.min_samples_leaf,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)

    # Save model
    joblib.dump(model, cfg.paths.model)

    return y_pred


if __name__ == "__main__":
    os.system("dvc fetch data/data.csv")
    os.system("dvc pull --remote myremote")
    train_model()
    os.system("dvc add model.joblib")
    os.system("dvc push model.joblib.dvc")
