"""
Custom exceptions for AI Evaluation Pipeline.
"""


class EvaluationPipelineError(Exception):
    """Base exception for all pipeline errors."""
    
    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DatasetError(EvaluationPipelineError):
    """Exception raised for dataset-related errors."""
    pass


class DatasetValidationError(DatasetError):
    """Exception raised when dataset validation fails."""
    pass


class ModelProviderError(EvaluationPipelineError):
    """Exception raised for model provider errors."""
    pass


class APIError(ModelProviderError):
    """Exception raised for API-related errors."""
    
    def __init__(
        self, 
        message: str, 
        status_code: int | None = None,
        provider: str | None = None,
        **kwargs
    ) -> None:
        super().__init__(message, **kwargs)
        self.status_code = status_code
        self.provider = provider


class RateLimitError(APIError):
    """Exception raised when API rate limit is exceeded."""
    
    def __init__(
        self, 
        message: str = "Rate limit exceeded",
        retry_after: float | None = None,
        **kwargs
    ) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class AuthenticationError(APIError):
    """Exception raised for authentication failures."""
    pass


class ModelNotFoundError(ModelProviderError):
    """Exception raised when a model is not found or not configured."""
    pass


class EvaluationError(EvaluationPipelineError):
    """Exception raised for evaluation-related errors."""
    pass


class MetricCalculationError(EvaluationError):
    """Exception raised when metric calculation fails."""
    pass


class StorageError(EvaluationPipelineError):
    """Exception raised for storage-related errors."""
    pass


class DatabaseConnectionError(StorageError):
    """Exception raised when database connection fails."""
    pass


class CacheError(EvaluationPipelineError):
    """Exception raised for caching-related errors."""
    pass


class ConfigurationError(EvaluationPipelineError):
    """Exception raised for configuration-related errors."""
    pass


class TimeoutError(EvaluationPipelineError):
    """Exception raised when an operation times out."""
    pass


class ConcurrencyError(EvaluationPipelineError):
    """Exception raised for concurrency-related errors."""
    pass