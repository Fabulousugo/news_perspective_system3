# Main configuration file
# CONFIG_SETTINGS = 
# config/settings.py


import os
from pathlib import Path
from typing import Dict, List
import yaml

class Settings:
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODEL_DIR = DATA_DIR / "models"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    
    # Model settings
    BIAS_CLASSIFIER_MODEL = "distilbert-base-uncased"
    SIMILARITY_MODEL = "all-MiniLM-L6-v2"
    MAX_SEQUENCE_LENGTH = 512
    BATCH_SIZE = 16
    
    # Political bias categories
    BIAS_LABELS = {
        0: "left-leaning",
        1: "centrist", 
        2: "right-leaning"
    }
    
    # API settings
    API_TIMEOUT = 30
    MAX_RETRIES = 3
    RATE_LIMIT_DELAY = 1
    
    # Processing settings
    MIN_ARTICLE_LENGTH = 100
    MAX_ARTICLES_PER_SOURCE = 1000
    SIMILARITY_THRESHOLD = 0.65  # Lowered from 0.75 for better recall
    
    @classmethod
    def load_news_sources(cls) -> Dict:
        with open(cls.PROJECT_ROOT / "config" / "news_sources.yaml", 'r') as f:
            return yaml.safe_load(f)
    
    @classmethod
    def get_api_key(cls, service: str) -> str:
        return os.getenv(f"{service.upper()}_API_KEY")

settings = Settings()