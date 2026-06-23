"""
Abstract base class for LLM providers and response models.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.core.exceptions import APIError, RateLimitError, TimeoutError as PipelineTimeoutError
from src.core.config import get_config

logger = logging.getLogger(__name__)


# =============================================================================
# Response Models
# =============================================================================


@dataclass
class LLMResponse:
    """
    Represents a response from an LLM provider.
    """
    model: str
    response_text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    raw_response: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    error: str | None = None
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.input_tokens + self.output_tokens
    
    @property
    def is_error(self) -> bool:
        """Check if the response contains an error."""
        return self.error is not None or not self.response_text
    
    @property
    def success(self) -> bool:
        """Check if the response was successful."""
        return not self.is_error


# =============================================================================
# Abstract Provider
# =============================================================================


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    All concrete implementations must inherit from this class.
    """
    
    provider_name: str = "base"
    
    def __init__(
        self,
        api_key: str | None = None,
        max_retries: int = 3,
        timeout: int = 60,
    ) -> None:
        """
        Initialize the LLM provider.
        
        Args:
            api_key: API key for authentication
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self._config = get_config()
        self._client: httpx.AsyncClient | None = None
    
    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        context: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The input prompt
            context: Optional context to include
            model: Model to use (uses default if not specified)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            LLMResponse object with the generated text and metadata
        """
        pass
    
    @abstractmethod
    def _get_headers(self) -> dict[str, str]:
        """Get the headers for API requests."""
        pass
    
    @abstractmethod
    def _get_base_url(self) -> str:
        """Get the base URL for API requests."""
        pass
    
    async def __aenter__(self) -> "LLMProvider":
        """Async context manager entry."""
        await self._ensure_client()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_client(self) -> None:
        """Ensure the HTTP client is initialized."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost for a model call."""
        return self._config.calculate_cost(model, input_tokens, output_tokens)
    
    def _handle_api_error(self, error: Exception, provider: str) -> None:
        """Handle and classify API errors."""
        error_str = str(error).lower()
        
        if "rate limit" in error_str or "429" in error_str:
            raise RateLimitError(
                f"Rate limit exceeded for {provider}",
                provider=provider,
            )
        elif "401" in error_str or "authentication" in error_str or "unauthorized" in error_str:
            raise APIError(
                f"Authentication failed for {provider}",
                status_code=401,
                provider=provider,
            )
        elif "404" in error_str or "not found" in error_str:
            raise APIError(
                f"Model not found for {provider}",
                status_code=404,
                provider=provider,
            )
        elif "timeout" in error_str:
            raise PipelineTimeoutError(f"Request timed out for {provider}")
        else:
            raise APIError(
                f"API error for {provider}: {error}",
                provider=provider,
            )


# =============================================================================
# Provider Factory
# =============================================================================


class LLMProviderFactory:
    """
    Factory class for creating LLM provider instances.
    """
    
    _providers: dict[str, type[LLMProvider]] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: type[LLMProvider]) -> None:
        """Register a provider class."""
        cls._providers[name.lower()] = provider_class
    
    @classmethod
    def create(
        cls,
        provider_name: str,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> LLMProvider:
        """
        Create a provider instance.
        
        Args:
            provider_name: Name of the provider (openai, anthropic, groq)
            api_key: API key for authentication
            **kwargs: Additional provider-specific arguments
            
        Returns:
            LLMProvider instance
            
        Raises:
            ValueError: If provider is not registered
        """
        provider_class = cls._providers.get(provider_name.lower())
        
        if provider_class is None:
            available = list(cls._providers.keys())
            raise ValueError(
                f"Unknown provider: {provider_name}. Available providers: {available}"
            )
        
        return provider_class(api_key=api_key, **kwargs)
    
    @classmethod
    def get_registered_providers(cls) -> list[str]:
        """Get list of registered provider names."""
        return list(cls._providers.keys())


def register_provider(name: str) -> type[LLMProvider]:
    """
    Decorator to register a provider class.
    
    Usage:
        @register_provider("openai")
        class OpenAIProvider(LLMProvider):
            ...
    """
    def decorator(cls: type[LLMProvider]) -> type[LLMProvider]:
        LLMProviderFactory.register(name, cls)
        return cls
    return decorator