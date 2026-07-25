import logging
from typing import Tuple
import pandas as pd
from src.data_cleaning import DataCleaner, DataPreProcessStrategy, DataDivideStrategy

def clean_data_step(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Step to preprocess and divide dataset into train/test sets using Strategy Pattern."""
    try:
        logging.info("Starting data preprocessing...")
        preprocess_strategy = DataPreProcessStrategy()
        cleaner = DataCleaner(df, preprocess_strategy)
        processed_data = cleaner.handle_data()

        logging.info("Splitting dataset into train/test sets...")
        divide_strategy = DataDivideStrategy(test_size=test_size, random_state=random_state)
        cleaner = DataCleaner(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = cleaner.handle_data()
        
        logging.info(f"Data cleaning & division complete. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in clean_data step: {e}")
        raise e
