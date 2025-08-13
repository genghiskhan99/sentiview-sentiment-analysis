import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Model configuration
    model_path: str = "./models/sentiment_lr_tfidf.pkl"
    
    # Amazon integration
    enable_amazon: bool = True
    amazon_dataset_url: str = "https://drive.google.com/uc?export=download&id=1SERc309kmcEGhsqhuztIE_ZaqJGku-WQ"
    
    # CORS configuration
    cors_origins: List[str] = ["*"]
    
    # Environment
    environment: str = "development"
    debug: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Amazon integration is enabled by default
        self.enable_amazon = True


settings = Settings()
