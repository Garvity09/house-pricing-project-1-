import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """Abstract Base Class for Model Evaluation Strategies."""
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

class MSE(Evaluation):
    """Strategy for Mean Squared Error."""
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"Mean Squared Error: {mse:.4f}")
            return float(mse)
        except Exception as e:
            logging.error(f"Error calculating MSE: {e}")
            raise e

class RMSE(Evaluation):
    """Strategy for Root Mean Squared Error."""
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info(f"Root Mean Squared Error: {rmse:.4f}")
            return float(rmse)
        except Exception as e:
            logging.error(f"Error calculating RMSE: {e}")
            raise e

class R2Score(Evaluation):
    """Strategy for R2 Score (Coefficient of Determination)."""
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2 Score: {r2:.4f}")
            return float(r2)
        except Exception as e:
            logging.error(f"Error calculating R2 Score: {e}")
            raise e
