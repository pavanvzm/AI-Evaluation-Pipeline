"""
Prompt Executor for AI Evaluation Pipeline.
Handles concurrent execution of prompts across multiple LLM providers.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.core.exceptions import (
    APIError,
    RateLimitError,
    TimeoutError as PipelineTimeoutError,
    ModelProviderError,
)
from src.core.config import get_config
from src.models.base import LLMProvider, LLMResponse, LLMProviderFactory
from src.evaluation.data_ingestion import Dataset, DatasetItem
from src.evaluation.metrics import EvaluationResult, MetricResult

logger = logging.getLogger(__name__)


# =============================================================================
# Execution Models
# =============================================================================


@dataclass
class ExecutionConfig:
    """Configuration for prompt execution."""
    max_concurrent: int = 5
    batch_size: int = 10
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    request_timeout: int = 60


@dataclass
class ExecutionProgress:
    """Tracks execution progress."""
    total: int = 0
    completed: int = 0
    failed: int = 0
    current_model: str = ""
    current_prompt_id: str = ""
    
    @property
    def percentage(self) -> float:
        """Get completion percentage."""
        if self.total == 0:
            return 0.0
        return (self.completed / self.total) * 100
    
    @property
    def success_rate(self) -> float:
        """Get success rate."""
        if self.completed + self.failed == 0:
            return 0.0
        return self.completed / (self.completed + self.failed)


# =============================================================================
# Prompt Executor
# =============================================================================


class PromptExecutor:
    """
    Executes prompts across multiple LLM providers with concurrency control.
    """
    
    def __init__(
        self,
        execution_config: ExecutionConfig | None = None,
        progress_callback: Callable[[ExecutionProgress], None] | None = None,
    ) -> None:
        """
        Initialize the prompt executor.
        
        Args:
            execution_config: Execution configuration
            progress_callback: Optional callback for progress updates
        """
        self._config = execution_config or ExecutionConfig()
        self._progress_callback = progress_callback
        self._config_manager = get_config()
        self._progress = ExecutionProgress()
        self._semaphore: asyncio.Semaphore | None = None
        self._lock = asyncio.Lock()
    
    async def execute_single(
        self,
        provider: LLMProvider,
        item: DatasetItem,
        model_name: str,
    ) -> EvaluationResult:
        """
        Execute a single prompt on a provider.
        
        Args:
            provider: LLM provider instance
            item: Dataset item with prompt and context
            model_name: Model name to use
            
        Returns:
            EvaluationResult
        """
        try:
            response = await self._call_with_retry(
                provider.generate_response,
                prompt=item.prompt,
                context=item.context,
                model=model_name,
            )
            
            return EvaluationResult(
                prompt_id=item.id,
                model_name=model_name,
                provider=provider.provider_name,
                response=response.response_text,
                latency_ms=response.latency_ms,
                cost_usd=response.cost_usd,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                ground_truth=item.ground_truth,
                context=item.context,
            )
            
        except RateLimitError as e:
            logger.warning(f"Rate limit hit for {model_name}: {e}")
            return self._create_error_result(item, model_name, provider.provider_name, str(e))
        except PipelineTimeoutError as e:
            logger.warning(f"Timeout for {model_name}: {e}")
            return self._create_error_result(item, model_name, provider.provider_name, str(e))
        except APIError as e:
            logger.warning(f"API error for {model_name}: {e}")
            return self._create_error_result(item, model_name, provider.provider_name, str(e))
        except Exception as e:
            logger.error(f"Unexpected error for {model_name}: {e}")
            return self._create_error_result(item, model_name, provider.provider_name, str(e))
    
    async def _call_with_retry(
        self,
        func: Callable[..., Coroutine],
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Call a function with exponential backoff retry.
        
        Args:
            func: Async function to call
            **kwargs: Arguments for the function
            
        Returns:
            LLMResponse from the function
        """
        delay = self._config.retry_base_delay
        last_error: Exception | None = None
        
        for attempt in range(self._config.max_retries):
            try:
                return await asyncio.wait_for(
                    func(**kwargs),
                    timeout=self._config.request_timeout,
                )
            except RateLimitError:
                # Don't retry rate limits immediately, wait longer
                delay = min(delay * 2, self._config.retry_max_delay)
                last_error = RateLimitError("Rate limit exceeded")
            except PipelineTimeoutError:
                last_error = PipelineTimeoutError("Request timed out")
            except APIError:
                last_error = APIError("API error")
            except asyncio.TimeoutError:
                last_error = PipelineTimeoutError("Request timed out")
            
            if attempt < self._config.max_retries - 1:
                logger.debug(f"Retry {attempt + 1}/{self._config.max_retries} after {delay}s")
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._config.retry_max_delay)
        
        raise last_error or APIError("Max retries exceeded")
    
    def _create_error_result(
        self,
        item: DatasetItem,
        model_name: str,
        provider: str,
        error: str,
    ) -> EvaluationResult:
        """Create an error result."""
        return EvaluationResult(
            prompt_id=item.id,
            model_name=model_name,
            provider=provider,
            response="",
            latency_ms=0.0,
            cost_usd=0.0,
            input_tokens=0,
            output_tokens=0,
            ground_truth=item.ground_truth,
            context=item.context,
            metrics={"error": MetricResult("error", 0.0, error=error)},
        )
    
    async def execute_dataset(
        self,
        dataset: Dataset,
        provider_name: str,
        model_name: str,
        api_key: str | None = None,
    ) -> list[EvaluationResult]:
        """
        Execute all prompts in a dataset with a single model.
        
        Args:
            dataset: Dataset to process
            provider_name: Name of the provider
            model_name: Model name to use
            api_key: Optional API key override
            
        Returns:
            List of evaluation results
        """
        results: list[EvaluationResult] = []
        self._progress = ExecutionProgress(
            total=len(dataset),
            current_model=model_name,
        )
        
        # Create provider
        provider = LLMProviderFactory.create(
            provider_name,
            api_key=api_key,
            timeout=self._config.request_timeout,
        )
        
        async with provider:
            # Create semaphore for concurrency control
            self._semaphore = asyncio.Semaphore(self._config.max_concurrent)
            
            async def execute_with_semaphore(item: DatasetItem) -> EvaluationResult:
                async with self._semaphore:
                    self._progress.current_prompt_id = item.id
                    result = await self.execute_single(provider, item, model_name)
                    
                    async with self._lock:
                        self._progress.completed += 1
                        if result.metrics.get("error"):
                            self._progress.failed += 1
                        
                        if self._progress_callback:
                            self._progress_callback(self._progress)
                    
                    return result
            
            # Execute all items
            tasks = [execute_with_semaphore(item) for item in dataset]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert exceptions to error results
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    final_results.append(
                        self._create_error_result(
                            dataset[i],
                            model_name,
                            provider_name,
                            str(result),
                        )
                    )
                else:
                    final_results.append(result)
        
        return final_results
    
    async def execute_multi_model(
        self,
        dataset: Dataset,
        models: list[tuple[str, str]],  # [(provider_name, model_name), ...]
        api_keys: dict[str, str] | None = None,
    ) -> dict[str, list[EvaluationResult]]:
        """
        Execute prompts across multiple models.
        
        Args:
            dataset: Dataset to process
            models: List of (provider_name, model_name) tuples
            api_keys: Optional API keys per provider
            
        Returns:
            Dictionary mapping model_name to results
        """
        api_keys = api_keys or {}
        all_results: dict[str, list[EvaluationResult]] = {}
        
        total_evaluations = len(dataset) * len(models)
        self._progress = ExecutionProgress(total=total_evaluations)
        
        for provider_name, model_name in models:
            logger.info(f"Evaluating {model_name} ({provider_name})")
            
            try:
                results = await self.execute_dataset(
                    dataset=dataset,
                    provider_name=provider_name,
                    model_name=model_name,
                    api_key=api_keys.get(provider_name),
                )
                all_results[model_name] = results
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                all_results[model_name] = [
                    self._create_error_result(item, model_name, provider_name, str(e))
                    for item in dataset
                ]
        
        return all_results
    
    def get_progress(self) -> ExecutionProgress:
        """Get current execution progress."""
        return self._progress


# =============================================================================
# Evaluation Runner
# =============================================================================


class EvaluationRunner:
    """
    High-level runner that combines execution and metric calculation.
    """
    
    def __init__(
        self,
        evaluator_provider: LLMProvider | None = None,
        execution_config: ExecutionConfig | None = None,
    ) -> None:
        """
        Initialize evaluation runner.
        
        Args:
            evaluator_provider: LLM provider for metric evaluation
            execution_config: Execution configuration
        """
        self._evaluator_provider = evaluator_provider
        self._executor = PromptExecutor(execution_config)
        self._config_manager = get_config()
    
    async def run(
        self,
        dataset: Dataset,
        models: list[tuple[str, str]],
        api_keys: dict[str, str] | None = None,
        calculate_metrics: bool = True,
    ) -> dict[str, list[EvaluationResult]]:
        """
        Run complete evaluation pipeline.
        
        Args:
            dataset: Dataset to evaluate
            models: List of (provider_name, model_name) tuples
            api_keys: Optional API keys per provider
            calculate_metrics: Whether to calculate quality metrics
            
        Returns:
            Dictionary mapping model_name to evaluation results
        """
        # Execute prompts
        results = await self._executor.execute_multi_model(
            dataset=dataset,
            models=models,
            api_keys=api_keys,
        )
        
        # Calculate metrics if requested
        if calculate_metrics:
            from src.evaluation.metrics import (
                AccuracyMetric,
                FaithfulnessMetric,
                HallucinationMetric,
            )
            
            accuracy_metric = AccuracyMetric()
            faithfulness_metric = FaithfulnessMetric(self._evaluator_provider)
            hallucination_metric = HallucinationMetric(self._evaluator_provider)
            
            for model_name, model_results in results.items():
                for result in model_results:
                    if result.response:  # Skip error results
                        # Calculate accuracy
                        if result.ground_truth:
                            acc_result = await accuracy_metric.calculate(
                                response=result.response,
                                ground_truth=result.ground_truth,
                                context=result.context,
                            )
                            result.metrics["accuracy"] = acc_result
                        
                        # Calculate faithfulness
                        if result.context:
                            faith_result = await faithfulness_metric.calculate(
                                response=result.response,
                                context=result.context,
                            )
                            result.metrics["faithfulness"] = faith_result
                            
                            # Calculate hallucination
                            hall_result = await hallucination_metric.calculate(
                                response=result.response,
                                context=result.context,
                            )
                            result.metrics["hallucination"] = hall_result
        
        return results
    
    def get_progress(self) -> ExecutionProgress:
        """Get current execution progress."""
        return self._executor.get_progress()