"""
Anthropic (Claude) provider implementation.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from src.core.exceptions import APIError, RateLimitError, TimeoutError as PipelineTimeoutError
from src.core.config import get_settings
from src.models.base import LLMProvider, LLMResponse, register_provider

logger = logging.getLogger(__name__)


@register_provider("anthropic")
class AnthropicProvider(LLMProvider):
    """
    Anthropic API provider for Claude models.
    """
    
    provider_name = "anthropic"
    BASE_URL = "https://api.anthropic.com/v1"
    
    def __init__(
        self,
        api_key: str | None = None,
        max_retries: int = 3,
        timeout: int = 60,
    ) -> None:
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
        super().__init__(api_key, max_retries, timeout)
        self._settings = get_settings()
        self._api_key = api_key or self._settings.anthropic_api_key
        
        if not self._api_key:
            logger.warning("Anthropic API key not provided")
    
    def _get_headers(self) -> dict[str, str]:
        """Get Anthropic API headers."""
        return {
            "x-api-key": self._api_key or "",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "anthropic-dangerous-direct-browser-access": "true",
        }
    
    def _get_base_url(self) -> str:
        """Get Anthropic API base URL."""
        return self.BASE_URL
    
    def _build_messages(
        self,
        prompt: str,
        context: str | None = None,
    ) -> str:
        """
        Build prompt for Anthropic messages API.
        
        Args:
            prompt: User prompt
            context: Optional context
            
        Returns:
            Formatted prompt string
        """
        if context:
            return f"\n\nContext:\n{context}\n\nPrompt: {prompt}"
        return prompt
    
    async def generate_response(
        self,
        prompt: str,
        context: str | None = None,
        model: str = "claude-3-5-sonnet-20240620",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """
        Generate a response from Anthropic Claude model.
        
        Args:
            prompt: User prompt
            context: Optional context
            model: Model name (claude-3-5-sonnet-20240620, claude-3-opus-20240229, etc.)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            LLMResponse object
        """
        start_time = time.time()
        
        await self._ensure_client()
        
        system_prompt = "You are a helpful assistant."
        if context:
            system_prompt = f"You are a helpful assistant. Use the following context to answer questions.\n\nContext:\n{context}"
        
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        try:
            response = await self._client.post(
                f"{self._get_base_url()}/messages",
                headers=self._get_headers(),
                json=payload,
            )
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 429:
                raise RateLimitError(
                    "Anthropic rate limit exceeded",
                    provider=self.provider_name,
                )
            
            if response.status_code == 401:
                raise APIError(
                    "Anthropic authentication failed",
                    status_code=401,
                    provider=self.provider_name,
                )
            
            if response.status_code != 200:
                error_data = response.json() if response.content else {}
                raise APIError(
                    f"Anthropic API error: {error_data.get('error', {}).get('message', response.text)}",
                    status_code=response.status_code,
                    provider=self.provider_name,
                )
            
            data = response.json()
            
            # Extract response data
            response_text = data["content"][0]["text"]
            usage = data.get("usage", {})
            
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            cost_usd = self._calculate_cost(model, input_tokens, output_tokens)
            
            return LLMResponse(
                model=model,
                response_text=response_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=elapsed_ms,
                cost_usd=cost_usd,
                raw_response=data,
                finish_reason=data.get("stop_reason"),
            )
            
        except RateLimitError:
            raise
        except APIError:
            raise
        except httpx.TimeoutException:
            raise PipelineTimeoutError(f"Request to Anthropic timed out after {self.timeout}s")
        except Exception as e:
            logger.error(f"Anthropic request failed: {e}")
            raise APIError(
                f"Anthropic request failed: {e}",
                provider=self.provider_name,
            )
    
    async def count_tokens(
        self,
        prompt: str,
        context: str | None = None,
        model: str = "claude-3-5-sonnet-20240620",
    ) -> int:
        """
        Count tokens in a prompt.
        
        Args:
            prompt: User prompt
            context: Optional context
            model: Model name
            
        Returns:
            Estimated token count
        """
        # Anthropic doesn't have a dedicated token counting endpoint
        # Fallback: rough estimation
        return len(prompt.split()) + len((context or "").split())