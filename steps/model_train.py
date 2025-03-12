import logging
import mlflow 
import pandas as pd
from zenml import step, pipeline
from sklearn.base import RegressorMixin
from src.model_dev import LinearRegressionModel
from pydantic import BaseModel

from zenml.client import Client


class ModelNameConfig(BaseModel):
    """
    Model configuration class.
    Allows specifying the model name for dynamic selection.
    """
    model_name: str = "LinearRegression"

    class Config:
        protected_namespaces = ()
        

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig = ModelNameConfig()
) -> RegressorMixin:
    """
    Trains the model on the provided training data.
    
    Args:
        X_train: Training features as a DataFrame.
        X_test: Testing features as a DataFrame.
        y_train: Training labels as a DataFrame.
        y_test: Testing labels as a DataFrame.
        config: Model configuration specifying the model name.

    Returns:
        The trained model instance.
    """
    try:
        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model {config.model_name} not supported.")
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e
