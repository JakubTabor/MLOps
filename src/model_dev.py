import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Model(ABC):
  """
  Abstract class forall models
  """
  
  @abstractmethod
  def train(self, X_train, y_train) -> RegressorMixin:
    """
    Trains the model
    Args:
        X_train: Training data
        y_train: Training labels
    Returns:
          None
    """
    pass
  
@abstractmethod
def predict(self, model: RegressorMixin, X_test):
  """
  Predicts the target values for the given test data.

  Args:
      model: Trained model instance.
      X_test: Test data.

  Returns:
      Predictions for the test data.
  """
  pass  
  
  
class LinearRegressionModel(Model):
  def train(self, X_train, y_train, **kwargs) -> LinearRegression:
    
    try:
      reg = LinearRegression(**kwargs)
      reg.fit(X_train, y_train)
      logging.info("Model training complete")
      return reg
    except Exception as e:
      logging.error("Error in training model: {}".format(e))
      raise 
    
  def predict(self, model: LinearRegression, X_test):
    """
    Predicts using the trained Linear Regression model.

    Args:
        model: Trained LinearRegression model.
        X_test: Test data.

    Returns:
        Predictions for the test data.
    """
    try:
      logging.info("Generating predictions...")
      predictions = model.predict(X_test)
      logging.info("Predictions complete.")
      return predictions
    except Exception as e:
      logging.error(f"Error in generating predictions: {e}")
      raise
