import numpy as np
import pandas as pd
import os
import logging
import json

from .utils import get_data_for_test
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from pydantic import BaseModel
import mlflow
from mlflow import set_experiment

from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Docker settings to include the required MLflow integration
docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseModel):
    """
    Configuration for the deployment trigger step.
    """
    min_accuracy: float = 0.92

@step(enable_cache=False)
def dynamic_importer() -> str:
    data = get_data_for_test()
    return data


@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig = DeploymentTriggerConfig(),
) -> bool:
    """
    Determines whether the model should be deployed based on accuracy.

    Args:
        accuracy: The accuracy of the model.
        config: Configuration for the deployment trigger.

    Returns:
        A boolean indicating whether the model meets the deployment criteria.
    """
    decision = accuracy >= config.min_accuracy
    if decision:
        logger.info(f"Model meets the accuracy threshold ({config.min_accuracy}). Deploying.")
    else:
        logger.info(f"Model accuracy ({accuracy}) is below the threshold ({config.min_accuracy}). Not deploying.")
    return decision


class MLFlowModelDeployerStepParameters(BaseModel):
    """
    Parameters for the MLflow model deployer step.
    """
    pipeline_name: str
    step_name: str
    running: bool = True


@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    
    mlflow_model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    services = mlflow_model_deployer.find_model_server(
        pipeline_name,
        pipeline_step_name,
        model_name,
        running=True
    )
    if not services:
        raise RuntimeError(f"No running services found for pipeline {pipeline_name}, step {pipeline_step_name}.")
    return services[0]

@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    
    service.start(timeout=DEFAULT_SERVICE_START_STOP_TIMEOUT)
    data = json.loads(data)
    data.pops("index")
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    predictions = service.predict(np.array(df.to_dict(orient="records")))
    return predictions

        
@pipeline(enable_cache=True, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0.92,
):

    if not os.path.exists(data_path):
        logger.error(f"The data path '{data_path}' does not exist.")
        raise FileNotFoundError(f"The data path '{data_path}' does not exist.")

    # Set MLflow URI and experiment
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    set_experiment("continuous_deployment_pipeline")

    logger.info("Starting the data ingestion step...")
    df = ingest_df(data_path=data_path)

    logger.info("Data cleaning and preprocessing...")
    X_train, X_test, y_train, y_test = clean_df(df)

    logger.info("Training the model...")
    model = train_model(X_train, X_test, y_train, y_test)

    logger.info("Evaluating the model...")
    r2_score, rmse = evaluate_model(model, X_test, y_test)
    logger.info(f"Evaluation metrics - R2 Score: {r2_score}, RMSE: {rmse}")

    logger.info("Checking deployment trigger conditions...")
    deployment_decision = deployment_trigger(accuracy=r2_score)

    logger.info("Running the model deployment step...")
    try:
        mlflow_model_deployer_step(
            model=model,
            deploy_decision=deployment_decision,
        )
        logger.info("Model deployment step completed successfully.")
    except Exception as e:
        logger.error(f"Error during model deployment: {e}")
        raise

    logger.info("Pipeline execution completed.")
    
    
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    data = dynamic_importer()
    service = prediction_service_loader(
            pipeline_name=pipeline_name,
            pipeline_step_name=pipeline_step_name,
            running=False,
    )
    prediction = predictor(service=service, data=data)
    return prediction
