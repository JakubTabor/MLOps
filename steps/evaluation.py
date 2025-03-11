import logging
import pandas as pd
import mlflow

from sklearn.base import RegressorMixin
from zenml import step
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from src.evaluation import MSE, RMSE, R2

from typing import Tuple
from typing_extensions import Annotated

# Retrieve the active experiment tracker from ZenML
client = Client()
experiment_tracker = client.active_stack.experiment_tracker

if experiment_tracker is None:
    logging.error("No experiment tracker found in the active ZenML stack. Please configure one.")
    raise ValueError("Experiment tracker is not configured in the active stack.")


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "rmse"],
]:
        """
        Evaluates the model on the test data and logs metrics to MLflow.

        Args:
            model: Trained model to evaluate.
            X_test: Features for testing.
            y_test: Labels for testing.

        Returns:
            r2_score: Coefficient of determination.
            rmse: Root Mean Squared Error.
        """
        if not hasattr(model, "predict"):
            logging.error("The provided model does not have a 'predict' method.")
            raise ValueError("Invalid model provided for evaluation.")

        if X_test.empty or y_test.empty:
            logging.error("Input data for evaluation is empty.")
            raise ValueError("X_test or y_test cannot be empty.")
        
        
        try:
            # Generate predictions
            predictions = model.predict(X_test)
            
            # Initialize metric calculators
            mse_class = MSE()
            r2_class = R2()
            rmse_class = RMSE()

            # Calculate metrics
            mse = mse_class.calculate_scores(y_test, predictions)
            r2 = r2_class.calculate_scores(y_test, predictions)
            rmse = rmse_class.calculate_scores(y_test, predictions)

            # Log metrics to MLflow
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("rmse", rmse)

            logging.info(f"Evaluation complete: R2={r2:.4f}, RMSE={rmse:.4f}")
            return r2, rmse

        except Exception as e:
            logging.error(f"Model evaluation failed: {e}")
            raise
