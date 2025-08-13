import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Model configuration
    model_path: str = "./models/sentiment_lr_tfidf.pkl"
    
    # Twitter integration
    enable_twitter: bool = False
    twitter_bearer_token: str = ""
    
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
        # Auto-enable Twitter if bearer token is provided
        if self.twitter_bearer_token and not self.enable_twitter:
            self.enable_twitter = True


settings = Settings()
