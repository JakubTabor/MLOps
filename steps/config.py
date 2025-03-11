from pydantic import BaseModel
class ModelNameConfig(BaseModel):
    """
    Model configs
    """
    model_name: str = "LinearRegression"
    class Config:
        protected_namespaces = ()
