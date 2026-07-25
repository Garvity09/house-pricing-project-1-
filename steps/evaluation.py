import logging
from typing import Tuple
import numpy as np
import pandas as pd

try:
    import mlflow
except ImportError:
    mlflow = None


from src.evaluation import MSE, RMSE, R2Score

def evaluation_step(model: any, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float, float]:
    """Step to evaluate model metrics (MSE, RMSE, R2) using Strategy Pattern."""
    try:
        logging.info("Starting model evaluation step...")
        predictions = model.predict(X_test)

        mse_strategy = MSE()
        mse = mse_strategy.calculate_scores(y_test, predictions)

        rmse_strategy = RMSE()
        rmse = rmse_strategy.calculate_scores(y_test, predictions)

        r2_strategy = R2Score()
        r2 = r2_strategy.calculate_scores(y_test, predictions)

        # MLflow Metric Logging
        if mlflow is not None:
            try:
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2_score", r2)
            except Exception as ml_err:
                logging.warning(f"MLflow metric logging warning: {ml_err}")


        logging.info(f"Evaluation complete -> MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2 Score: {r2:.4f}")
        return mse, rmse, r2
    except Exception as e:
        logging.error(f"Error in evaluation step: {e}")
        raise e
