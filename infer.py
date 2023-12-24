import os

import hydra
import joblib
import pandas as pd


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run_inference(cfg):
    # Load model
    model = joblib.load(cfg.model_path)

    # Load input data
    X_test = pd.read_csv(cfg.X_test_path)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.to_csv(cfg.predicted_path, index=False)
    # return predictions
    return y_pred


if __name__ == "__main__":
    os.system("dvc fetch model.joblib")
    os.system("dvc fetch X_test.csv")
    os.system("dvc pull --remote myremote")
    run_inference()
    os.system("dvc add predicted.csv")
    os.system("dvc push predicted.csv.dvc")
