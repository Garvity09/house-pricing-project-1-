import logging
import pandas as pd
from src.data_ingestion import DataIngestorFactory

def ingest_data_step(file_path: str) -> pd.DataFrame:
    """Step to ingest data from zip or csv using Factory Pattern."""
    try:
        logging.info(f"Starting data ingestion step for: {file_path}")
        ingestor = DataIngestorFactory.get_data_ingestor(file_path)
        df = ingestor.get_data(file_path)
        logging.info(f"Ingestion step complete. Loaded dataset with shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error in ingest_data step: {e}")
        raise e
