from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.model_train import train_model 


@pipeline(enable_cache=True)
def train_pipeline(data_path: str):
   df = ingest_df(data_path)
   X_train, X_test, y_train, y_test = clean_df(df)
   model = train_model(X_train, X_test,y_train, y_test)
   r2, rmse = evaluate_model(model, X_test, y_test) 
