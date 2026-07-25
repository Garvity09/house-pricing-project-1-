import os
import joblib
import logging
from typing import Any
import pandas as pd

try:
    import mlflow
except ImportError:
    mlflow = None


from src.model_dev import LinearRegressionModel, RandomForestModel, XGBoostModel, LightGBMModel

def train_model_step(X_train: pd.DataFrame, y_train: pd.Series, model_name: str = "lightgbm") -> Any:
    """Step to train specified machine learning model and log with MLflow."""
    try:
        model_name = model_name.lower()
        logging.info(f"Starting model training step for strategy: {model_name}")

        if model_name == "linear_regression":
            model_strategy = LinearRegressionModel()
        elif model_name == "random_forest":
            model_strategy = RandomForestModel()
        elif model_name == "xgboost":
            model_strategy = XGBoostModel()
        elif model_name == "lightgbm":
            model_strategy = LightGBMModel()
        else:
            raise ValueError(f"Model strategy '{model_name}' not supported.")

        model = model_strategy.train(X_train, y_train)

        # Local artifact caching
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", f"{model_name}_model.pkl")
        joblib.dump(model, model_path)
        logging.info(f"Saved trained model locally to {model_path}")

        # MLflow Logging
        if mlflow is not None:
            try:
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("num_features", X_train.shape[1])
                mlflow.log_param("train_samples", X_train.shape[0])
                mlflow.sklearn.log_model(model, artifact_path="model")
            except Exception as ml_err:
                logging.warning(f"MLflow tracking warning: {ml_err}")


        return model
    except Exception as e:
        logging.error(f"Error in model_train step: {e}")
        raise e
