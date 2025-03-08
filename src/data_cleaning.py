import logging
import os
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
  """
  Abctract class defining strategy for handling data
  """
  
  
  @abstractmethod
  def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
    pass
  
class DataPreProcessStrategy(DataStrategy):
  """
  Strategy for preprocessing data
  """
  
  def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess data
    """
    try: 
        cols_to_drop = [
              "order_approved_at",
              "order_delivered_carrier_date",
              "order_delivered_customer_date",
              "order_estimated_delivery_date",
              "order_purchase_timestamp",
        ]
        existing_cols = [col for col in cols_to_drop if col in data.columns]
        data = data.drop(existing_cols, axis=1)
          
        
        # Fill missing values
        data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
        data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
        data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
        data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
        data["review_comment_message"].fillna("No review", inplace=True)
        
        # Keep only numeric columns and drop specified columns
        data = data.select_dtypes(include=[np.number])
        cols_to_drop_numeric = ["customer_zip_code_prefix", "order_item_id"]
        existing_numeric_columns = [col for col in cols_to_drop_numeric if col in data.columns]
        data = data.drop(existing_numeric_columns, axis=1)
        
        return data
    
    
    except Exception as e:
      logging.error("Error in processing data: {}".format(e))
      raise e
   
    
class DataDivideStrategy(DataStrategy):
  """
  Strategy for dividing data into train test plit
  """
  
  def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
    """
    Divide data into train and test
    """
    
    try:
      X = data.drop(["review_score"], axis=1)
      y = data["review_score"]
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
      return X_train, X_test, y_train, y_test
    except Exception as e:
      logging.error("Error in dividing data: {}".format(e))
      raise e
    
    
class DataCleaning: 
  """
  Class for cleaning data which processes the data and divide in into train and test
  """ 
  
  def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
    self.data = data
    self.strategy = strategy
    
  def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
    """
    Handle data
    """
    try: 
       return self.strategy.handle_data(self.data)
    except Exception as e:
      logging.error("Error in handling data: {}".format(e))
      raise e

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    file_path = os.path.join(os.getcwd(), "data/olist_customer_dataset.csv")
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    # Load data
    logging.info(f"Loading data from {file_path}")
    data = pd.read_csv(file_path)

    # Preprocess data
    logging.info("Starting data preprocessing...")
    data_preprocessor = DataCleaning(data, DataPreProcessStrategy())
    processed_data = data_preprocessor.handle_data()
    logging.info("Data preprocessing completed.")

    # Save processed data
    processed_file_path = os.path.join(os.getcwd(), "data/processed_data.csv")
    processed_data.to_csv(processed_file_path, index=False)
    logging.info(f"Processed data saved to {processed_file_path}")

    # Split data into training and testing sets
    logging.info("Splitting data into training and testing sets...")
    data_splitter = DataCleaning(processed_data, DataDivideStrategy())
    X_train, X_test, y_train, y_test = data_splitter.handle_data()
    logging.info("Data splitting completed.")

    # Print shapes of splits for verification
    logging.info(f"Training data shape: {X_train.shape}")
    logging.info(f"Test data shape: {X_test.shape}")
