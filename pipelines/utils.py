import logging
import os
import pandas as pd

from src.data_cleaning import DataCleaning, DataPreProcessStrategy

def get_data_for_test():
  try:
        # Define the dataset path
        file_path = os.path.join("data", "olist_customers_dataset.csv")
        logging.info(f"Reading data from {file_path}...")

        # Load the dataset
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")

        # Sample 100 rows for testing
        logging.info("Sampling 100 rows from the dataset...")
        df = df.sample(n=100)

        # Apply data preprocessing and cleaning
        logging.info("Applying data preprocessing and cleaning...")
        preprocess_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        df = data_cleaning.handle_data()

        # Drop the 'review_score' column
        logging.info("Dropping 'review_score' column...")
        df.drop(["review_score"], axis=1, inplace=True)

        # Serialize DataFrame to JSON
        logging.info("Serializing processed DataFrame to JSON...")
        result = df.to_json(orient="split")
        logging.info("Data serialization completed successfully.")

        return result
  
  except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
  except pd.errors.EmptyDataError as e:
      logging.error(f"Empty dataset error: {e}")
      raise
  except Exception as e:
      logging.error(f"An error occurred during data processing: {e}")
      raise
