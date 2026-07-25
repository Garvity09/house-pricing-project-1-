import logging
from typing import Tuple, Dict, Any

from steps.ingest_data import ingest_data_step
from steps.clean_data import clean_data_step
from steps.model_train import train_model_step
from steps.evaluation import evaluation_step

def train_pipeline(data_path: str, model_name: str = "lightgbm") -> Dict[str, Any]:
    """Training Pipeline connecting data ingestion, cleaning, model training, and evaluation steps."""
    logging.info("================ Running Customer Satisfaction Pipeline ================")
    
    # 1. Data Ingestion Step (Factory Pattern)
    df = ingest_data_step(file_path=data_path)
    
    # 2. Data Cleaning & Splitting Step (Strategy Pattern)
    X_train, X_test, y_train, y_test = clean_data_step(df=df)
    
    # 3. Model Training Step (Strategy Pattern)
    model = train_model_step(X_train=X_train, y_train=y_train, model_name=model_name)
    
    # 4. Evaluation Step (Strategy Pattern)
    mse, rmse, r2 = evaluation_step(model=model, X_test=X_test, y_test=y_test)
    
    pipeline_result = {
        "model": model,
        "model_name": model_name,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "metrics": {
            "mse": mse,
            "rmse": rmse,
            "r2_score": r2
        }
    }
    logging.info("================ Pipeline Execution Finished Successfully ================")
    return pipeline_result
