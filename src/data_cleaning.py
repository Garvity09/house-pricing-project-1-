import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """Abstract Base Class for Data Strategy design pattern."""
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        pass

class DataPreProcessStrategy(DataStrategy):
    """Data Preprocessing Strategy: cleans dataset, selects relevant features, handles missing values."""
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            df = data.copy()
            
            # Fill or drop missing values
            df = df.dropna()

            # Feature Engineering: volume calculation if spatial dimensions present
            if all(col in df.columns for col in ["product_length_cm", "product_height_cm", "product_width_cm"]):
                df["product_volume_cm3"] = df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
            
            # Drop non-predictive identifier columns if present
            cols_to_drop = ["order_id", "customer_id"]
            df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
            
            return df
        except Exception as e:
            logging.error(f"Error in data preprocessing: {e}")
            raise e

class DataDivideStrategy(DataStrategy):
    """Data Divide Strategy: splits dataset into training and testing sets."""
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def handle_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        try:
            if "review_score" not in data.columns:
                raise KeyError("'review_score' column missing from dataset.")
            
            X = data.drop(columns=["review_score"])
            y = data["review_score"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in splitting data: {e}")
            raise e

class DataCleaner:
    """Context class using a DataStrategy."""
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        return self.strategy.handle_data(self.data)
