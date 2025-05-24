"""
Configuration management for GFRIP
Handles loading and validation of configuration settings
"""

import os
import yaml
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field, validator
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataSourceConfig(BaseSettings):
    """Configuration for a data source"""
    name: str
    type: str  # e.g., 'api', 'database', 'file'
    enabled: bool = True
    priority: int = 10
    params: Dict[str, Any] = {}
    
    class Config:
        extra = 'forbid'  # Prevent extra fields

class ModelConfig(BaseSettings):
    """Configuration for a machine learning model"""
    name: str
    type: str  # e.g., 'gnn', 'transformer', 'classifier'
    path: str
    version: str = "1.0.0"
    params: Dict[str, Any] = {}
    enabled: bool = True
    
    class Config:
        extra = 'forbid'

class APIConfig(BaseSettings):
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 1
    secret_key: str = "your-secret-key-here"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    class Config:
        extra = 'forbid'

class LoggingConfig(BaseSettings):
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    max_size_mb: int = 100
    backup_count: int = 5
    
    class Config:
        extra = 'forbid'

class GFRIPConfig(BaseSettings):
    """Main configuration class for GFRIP"""
    environment: str = "development"
    data_sources: Dict[str, DataSourceConfig] = {}
    models: Dict[str, ModelConfig] = {}
    api: APIConfig = APIConfig()
    logging: LoggingConfig = LoggingConfig()
    
    class Config:
        extra = 'forbid'
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'GFRIPConfig':
        """Load configuration from a YAML file"""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}
            return cls(**config_data)
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return self.dict()
    
    def save(self, config_path: str) -> None:
        """Save configuration to a YAML file"""
        try:
            with open(config_path, 'w') as f:
                yaml.dump(self.dict(), f, default_flow_style=False)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {e}")
            raise

# Default configuration paths
DEFAULT_CONFIG_PATHS = [
    Path("config/config.yaml"),
    Path("config/default.yaml"),
    Path("gfrip/config/config.yaml"),
    Path("gfrip/config/default.yaml"),
    Path("/etc/gfrip/config.yaml"),
]

def load_config(config_path: Optional[str] = None) -> GFRIPConfig:
    """
    Load configuration from file or use defaults
    
    Args:
        config_path: Optional path to config file. If None, search default locations.
        
    Returns:
        GFRIPConfig: Loaded configuration
    """
    # If config path is provided, try to load from there first
    if config_path:
        try:
            return GFRIPConfig.from_yaml(config_path)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
    
    # Otherwise, try default locations
    for path in DEFAULT_CONFIG_PATHS:
        if path.exists():
            try:
                return GFRIPConfig.from_yaml(str(path))
            except Exception as e:
                logger.warning(f"Failed to load config from {path}: {e}")
    
    # If no config file found, use defaults
    logger.warning("No configuration file found, using default settings")
    return GFRIPConfig()

def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """
    Configure logging based on configuration
    
    Args:
        config: Optional logging configuration. If None, uses default.
    """
    if config is None:
        config = LoggingConfig()
    
    # Set basic config
    logging.basicConfig(
        level=getattr(logging, config.level, logging.INFO),
        format=config.format,
        force=True
    )
    
    # Add file handler if specified
    if config.file:
        from logging.handlers import RotatingFileHandler
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(config.file)), exist_ok=True)
        
        file_handler = RotatingFileHandler(
            config.file,
            maxBytes=config.max_size_mb * 1024 * 1024,
            backupCount=config.backup_count
        )
        file_handler.setFormatter(logging.Formatter(config.format))
        logging.getLogger().addHandler(file_handler)
    
    logger.info(f"Logging configured with level {config.level}")

# Example usage
if __name__ == "__main__":
    # Load config
    config = load_config()
    
    # Setup logging
    setup_logging(config.logging)
    
    # Example of using the config
    logger.info(f"Starting GFRIP in {config.environment} mode")
    logger.info(f"API will run on {config.api.host}:{config.api.port}")
