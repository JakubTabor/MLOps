import logging
from pydantic import BaseModel
import pandas as pd
from zenml import step

class IngestDataConfig(BaseModel):
    """
    Configuration for ingesting data.
    """
    data_path: str = "/path/to/your/dataset.csv"

class IngestData:
    """
    Handles data ingestion from a specified file path.
    """
    def __init__(self, data_path: str):
        """
        Args:
            data_path: Path to the data file.
        """
        self.data_path = data_path

    def get_data(self) -> pd.DataFrame:
        """
        Reads the data from the file path.

        Returns:
            pd.DataFrame: The ingested data.
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def ingest_df(data_path: str) -> pd.DataFrame:
  """
  Ingesting the data from the data_path.
  
  Args:
      data_paths: path to the data
  Returns:
      pd.DataFrame: the ingest data 
  """
  try:
        logging.info("Starting the data ingestion step...")
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        logging.info("Data ingestion step completed successfully.")
        return df
  except Exception as e:
    logging.error(f"Data ingestion failed: {e}")
    raise
