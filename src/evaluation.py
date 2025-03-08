import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Evaluation(ABC):
  """
  Abstract class defining strategy for evaluation our models
  """
  @abstractmethod
  def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
      """
      Calculate the evaluation metric scores for the model.
      
      Args:
          y_true: True labels.
          y_pred: Predicted labels.

      Returns:
          The calculated metric score.
      """
      pass
  
  
class MSE(Evaluation):
    """
    Evaluation strategy that uses Mean Squared Error (MSE).
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculating MSE...")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in calculating MSE: {e}")
            raise
    
    
class R2(Evaluation):
    """
    Evaluation strategy that uses R-squared (R²) Score.
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculating R2 Score...")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2 Score: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error in calculating R2 Score: {e}")
            raise
    
    
class RMSE(Evaluation):
  """
  Evaluation strategy that uses Root Mean Squared Error
  """
  def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        logging.info("Calculating RMSE...")
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        logging.info(f"RMSE: {rmse}")
        return rmse
    except Exception as e:
        logging.error(f"Error in calculating RMSE: {e}")
        raise
