"""
Configuration management for AI Evaluation Pipeline.
Loads settings from YAML files and environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# =============================================================================
# Configuration Models
# =============================================================================


class ModelPricing(BaseModel):
    """Pricing configuration for a model."""
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0


class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    name: str
    display_name: str
    max_tokens: int = 4096
    temperature: float = 0.7
    pricing: ModelPricing = Field(default_factory=ModelPricing)


class ProviderConfig(BaseModel):
    """Configuration for a model provider."""
    provider: str
    models: list[ModelConfig] = Field(default_factory=list)


class EvaluationConfig(BaseModel):
    """Configuration for evaluation settings."""
    max_concurrent_requests: int = 5
    batch_size: int = 10
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    retry_multiplier: float = 2.0
    request_timeout: int = 60
    metric_weights: dict[str, float] = Field(default_factory=lambda: {
        "accuracy": 0.30,
        "faithfulness": 0.25,
        "hallucination": 0.25,
        "latency": 0.10,
        "cost": 0.10
    })


class StorageConfig(BaseModel):
    """Configuration for storage settings."""
    database: dict[str, str] = Field(default_factory=lambda: {
        "type": "sqlite",
        "path": "data/evaluation_results.db"
    })
    results_export: dict[str, Any] = Field(default_factory=lambda: {
        "formats": ["csv", "json"],
        "directory": "data/results"
    })


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/evaluation.log"


class DashboardConfig(BaseModel):
    """Configuration for dashboard settings."""
    host: str = "0.0.0.0"
    port: int = 8501
    theme: str = "light"


class AppConfig(BaseModel):
    """Root configuration model."""
    models: dict[str, ProviderConfig] = Field(default_factory=dict)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)


class Settings(BaseSettings):
    """Environment variables settings."""
    # API Keys
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    together_api_key: str | None = Field(default=None, alias="TOGETHER_API_KEY")
    
    # Azure OpenAI
    azure_openai_api_key: str | None = Field(default=None, alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str | None = Field(default=None, alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_deployment_name: str | None = Field(default=None, alias="AZURE_OPENAI_DEPLOYMENT_NAME")
    azure_openai_api_version: str | None = Field(default=None, alias="AZURE_OPENAI_API_VERSION")
    
    # Evaluation Settings
    max_concurrent_requests: int = Field(default=5, alias="MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(default=60, alias="REQUEST_TIMEOUT")
    enable_caching: bool = Field(default=True, alias="ENABLE_CACHING")
    
    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: str = Field(default="logs/evaluation.log", alias="LOG_FILE")
    
    # Dashboard
    dashboard_port: int = Field(default=8501, alias="DASHBOARD_PORT")
    real_time_updates: bool = Field(default=True, alias="REAL_TIME_UPDATES")
    
    # Evaluator Model
    evaluator_model: str = Field(default="gpt-4o-mini", alias="EVALUATOR_MODEL")
    
    # Embedding Model
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# =============================================================================
# Configuration Manager
# =============================================================================


class ConfigManager:
    """
    Manages application configuration from YAML files and environment variables.
    Singleton pattern to ensure consistent configuration across the application.
    """
    
    _instance: ConfigManager | None = None
    _config: AppConfig | None = None
    _settings: Settings | None = None
    
    def __new__(cls) -> ConfigManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        if self._config is None:
            self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file and environment variables."""
        # Load YAML configuration
        config_path = self._find_config_file()
        if config_path and config_path.exists():
            with open(config_path, "r") as f:
                yaml_data = yaml.safe_load(f)
                self._config = AppConfig(**yaml_data)
        else:
            self._config = AppConfig()
        
        # Load environment settings
        self._settings = Settings()
    
    def _find_config_file(self) -> Path | None:
        """Find the configuration file in standard locations."""
        possible_paths = [
            Path("config/config.yaml"),
            Path(__file__).parent.parent.parent / "config" / "config.yaml",
            Path.cwd() / "config" / "config.yaml",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        return None
    
    @property
    def config(self) -> AppConfig:
        """Get the main configuration object."""
        return self._config or AppConfig()
    
    @property
    def settings(self) -> Settings:
        """Get the environment settings object."""
        return self._settings or Settings()
    
    def get_api_key(self, provider: str) -> str | None:
        """Get API key for a specific provider from environment variables."""
        key_map = {
            "openai": self.settings.openai_api_key,
            "anthropic": self.settings.anthropic_api_key,
            "groq": self.settings.groq_api_key,
            "together": self.settings.together_api_key,
            "azure": self.settings.azure_openai_api_key,
        }
        return key_map.get(provider.lower())
    
    def get_model_config(self, model_name: str) -> ModelConfig | None:
        """Get configuration for a specific model by name."""
        for provider_config in self.config.models.values():
            for model in provider_config.models:
                if model.name == model_name:
                    return model
        return None
    
    def get_all_models(self) -> list[tuple[str, ModelConfig]]:
        """Get all configured models with their provider names."""
        models = []
        for provider_name, provider_config in self.config.models.items():
            for model in provider_config.models:
                models.append((provider_name, model))
        return models
    
    def calculate_cost(
        self, 
        model_name: str, 
        input_tokens: int, 
        output_tokens: int
    ) -> float:
        """Calculate the cost for a given model and token usage."""
        model_config = self.get_model_config(model_name)
        if model_config is None:
            return 0.0
        
        input_cost = (input_tokens / 1000) * model_config.pricing.input_cost_per_1k
        output_cost = (output_tokens / 1000) * model_config.pricing.output_cost_per_1k
        return round(input_cost + output_cost, 6)
    
    def reload(self) -> None:
        """Reload configuration from files and environment."""
        self._load_config()


# Global configuration instance
def get_config() -> ConfigManager:
    """Get the global configuration manager instance."""
    return ConfigManager()


# Convenience function for quick access
def get_settings() -> Settings:
    """Get the environment settings."""
    return get_config().settings


def get_model_config(model_name: str) -> ModelConfig | None:
    """Get configuration for a specific model."""
    return get_config().get_model_config(model_name)


def get_all_models() -> list[tuple[str, ModelConfig]]:
    """Get all configured models."""
    return get_config().get_all_models()


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for a model call."""
    return get_config().calculate_cost(model_name, input_tokens, output_tokens)