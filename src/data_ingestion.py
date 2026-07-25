import os
import zipfile
from abc import ABC, abstractmethod
import pandas as pd

class DataIngestor(ABC):
    """Abstract Base Class for Data Ingestion strategies."""
    @abstractmethod
    def get_data(self, file_path: str) -> pd.DataFrame:
        pass

class ZipDataIngestor(DataIngestor):
    """Data Ingestor for .zip archives containing tabular CSV data."""
    def get_data(self, file_path: str) -> pd.DataFrame:
        if not file_path.endswith(".zip"):
            raise ValueError("Extracting zip file requires a .zip extension.")
        
        extract_dir = os.path.dirname(file_path)
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
            extracted_files = [f for f in zip_ref.namelist() if f.endswith(".csv")]
        
        if not extracted_files:
            raise FileNotFoundError("No CSV file found inside the zip archive.")
        
        csv_file_path = os.path.join(extract_dir, extracted_files[0])
        print(f"[ZipDataIngestor] Successfully extracted and loading: {csv_file_path}")
        return pd.read_csv(csv_file_path)

class CSVDataIngestor(DataIngestor):
    """Data Ingestor for standalone .csv files."""
    def get_data(self, file_path: str) -> pd.DataFrame:
        if not file_path.endswith(".csv"):
            raise ValueError("CSV Ingestor requires a .csv file extension.")
        print(f"[CSVDataIngestor] Successfully loading CSV file: {file_path}")
        return pd.read_csv(file_path)

class DataIngestorFactory:
    """Factory Design Pattern for creating Data Ingestors."""
    @staticmethod
    def get_data_ingestor(file_path: str) -> DataIngestor:
        if file_path.endswith(".zip"):
            return ZipDataIngestor()
        elif file_path.endswith(".csv"):
            return CSVDataIngestor()
        else:
            raise ValueError(f"No DataIngestor available for file format: {file_path}")

def ingest_data(file_path: str) -> pd.DataFrame:
    """Convenience function using the DataIngestorFactory."""
    ingestor = DataIngestorFactory.get_data_ingestor(file_path)
    return ingestor.get_data(file_path)
