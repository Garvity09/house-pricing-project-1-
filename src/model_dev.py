import logging
from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None


class Model(ABC):
    """Abstract Base Class for Machine Learning Models using Strategy Pattern."""
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        pass

class LinearRegressionModel(Model):
    """Linear Regression Strategy implementation."""
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
        try:
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            logging.info("Linear Regression Model trained successfully.")
            return reg
        except Exception as e:
            logging.error(f"Error training Linear Regression: {e}")
            raise e

class RandomForestModel(Model):
    """Random Forest Regressor Strategy implementation."""
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
        try:
            reg = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            reg.fit(X_train, y_train)
            logging.info("Random Forest Model trained successfully.")
            return reg
        except Exception as e:
            logging.error(f"Error training Random Forest: {e}")
            raise e

class XGBoostModel(Model):
    """XGBoost Regressor Strategy implementation."""
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        try:
            if XGBRegressor is not None:
                reg = XGBRegressor(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state
                )
                reg.fit(X_train, y_train)
                logging.info("XGBoost Model trained successfully.")
            else:
                logging.warning("XGBoost package not installed. Falling back to RandomForestRegressor.")
                reg = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
                reg.fit(X_train, y_train)
            return reg
        except Exception as e:
            logging.error(f"Error training XGBoost: {e}")
            raise e

class LightGBMModel(Model):
    """LightGBM Regressor Strategy implementation."""
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        try:
            if LGBMRegressor is not None:
                reg = LGBMRegressor(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                    verbose=-1
                )
                reg.fit(X_train, y_train)
                logging.info("LightGBM Model trained successfully.")
            else:
                logging.warning("LightGBM package not installed. Falling back to RandomForestRegressor.")
                reg = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
                reg.fit(X_train, y_train)
            return reg
        except Exception as e:
            logging.error(f"Error training LightGBM: {e}")
            raise e

