"""
OpenAI (GPT) provider implementation.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from src.core.exceptions import APIError, RateLimitError, TimeoutError as PipelineTimeoutError
from src.core.config import get_config, get_settings
from src.models.base import LLMProvider, LLMResponse, register_provider

logger = logging.getLogger(__name__)


@register_provider("openai")
class OpenAIProvider(LLMProvider):
    """
    OpenAI API provider for GPT models.
    """
    
    provider_name = "openai"
    BASE_URL = "https://api.openai.com/v1"
    
    def __init__(
        self,
        api_key: str | None = None,
        max_retries: int = 3,
        timeout: int = 60,
    ) -> None:
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
        super().__init__(api_key, max_retries, timeout)
        self._settings = get_settings()
        self._api_key = api_key or self._settings.openai_api_key
        
        if not self._api_key:
            logger.warning("OpenAI API key not provided")
    
    def _get_headers(self) -> dict[str, str]:
        """Get OpenAI API headers."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
    
    def _get_base_url(self) -> str:
        """Get OpenAI API base URL."""
        return self.BASE_URL
    
    def _build_messages(
        self,
        prompt: str,
        context: str | None = None,
    ) -> list[dict[str, str]]:
        """
        Build message list for chat completions API.
        
        Args:
            prompt: User prompt
            context: Optional context
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # Add system message with context if provided
        if context:
            messages.append({
                "role": "system",
                "content": f"You are a helpful assistant. Use the following context to answer questions.\n\nContext:\n{context}"
            })
        else:
            messages.append({
                "role": "system",
                "content": "You are a helpful assistant."
            })
        
        # Add user prompt
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        return messages
    
    async def generate_response(
        self,
        prompt: str,
        context: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """
        Generate a response from OpenAI GPT model.
        
        Args:
            prompt: User prompt
            context: Optional context
            model: Model name (gpt-4o, gpt-4o-mini, gpt-4-turbo, etc.)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            LLMResponse object
        """
        start_time = time.time()
        
        await self._ensure_client()
        
        messages = self._build_messages(prompt, context)
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            response = await self._client.post(
                f"{self._get_base_url()}/chat/completions",
                headers=self._get_headers(),
                json=payload,
            )
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 429:
                raise RateLimitError(
                    "OpenAI rate limit exceeded",
                    provider=self.provider_name,
                )
            
            if response.status_code == 401:
                raise APIError(
                    "OpenAI authentication failed",
                    status_code=401,
                    provider=self.provider_name,
                )
            
            if response.status_code != 200:
                error_data = response.json() if response.content else {}
                raise APIError(
                    f"OpenAI API error: {error_data.get('error', {}).get('message', response.text)}",
                    status_code=response.status_code,
                    provider=self.provider_name,
                )
            
            data = response.json()
            
            # Extract response data
            choice = data["choices"][0]
            response_text = choice["message"]["content"]
            usage = data.get("usage", {})
            
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            cost_usd = self._calculate_cost(model, input_tokens, output_tokens)
            
            return LLMResponse(
                model=model,
                response_text=response_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=elapsed_ms,
                cost_usd=cost_usd,
                raw_response=data,
                finish_reason=choice.get("finish_reason"),
            )
            
        except RateLimitError:
            raise
        except APIError:
            raise
        except httpx.TimeoutException:
            raise PipelineTimeoutError(f"Request to OpenAI timed out after {self.timeout}s")
        except Exception as e:
            logger.error(f"OpenAI request failed: {e}")
            raise APIError(
                f"OpenAI request failed: {e}",
                provider=self.provider_name,
            )
    
    async def count_tokens(
        self,
        prompt: str,
        context: str | None = None,
        model: str = "gpt-4o-mini",
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
        await self._ensure_client()
        
        messages = self._build_messages(prompt, context)
        
        try:
            response = await self._client.post(
                f"{self._get_base_url()}/chat/completions",
                headers=self._get_headers(),
                json={"model": model, "messages": messages},
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("usage", {}).get("prompt_tokens", 0)
            return 0
            
        except Exception:
            # Fallback: rough estimation
            return len(prompt.split()) + len((context or "").split())