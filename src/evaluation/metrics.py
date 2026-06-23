"""
Evaluation Engine for AI Evaluation Pipeline.
Implements quality and performance metrics for LLM evaluation.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# =============================================================================
# Metric Result Models
# =============================================================================


@dataclass
class MetricResult:
    """Result of a metric calculation."""
    name: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    
    @property
    def success(self) -> bool:
        """Check if the metric calculation was successful."""
        return self.error is None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "score": self.score,
            "details": self.details,
            "error": self.error,
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result for a single prompt-model pair."""
    prompt_id: str
    model_name: str
    provider: str
    response: str
    latency_ms: float
    cost_usd: float
    input_tokens: int
    output_tokens: int
    metrics: dict[str, MetricResult] = field(default_factory=dict)
    ground_truth: str | None = None
    context: str | None = None
    
    @property
    def accuracy(self) -> float:
        """Get accuracy score."""
        return self.metrics.get("accuracy", MetricResult("accuracy", 0.0)).score
    
    @property
    def faithfulness(self) -> float:
        """Get faithfulness score."""
        return self.metrics.get("faithfulness", MetricResult("faithfulness", 0.0)).score
    
    @property
    def hallucination_score(self) -> float:
        """Get hallucination score (1 - faithfulness)."""
        return 1.0 - self.faithfulness
    
    @property
    def composite_score(self) -> float:
        """Get weighted composite score."""
        weights = {
            "accuracy": 0.30,
            "faithfulness": 0.25,
            "hallucination": 0.25,
            "latency": 0.10,
            "cost": 0.10,
        }
        
        # Normalize latency and cost (higher is worse)
        latency_score = max(0, 1.0 - min(self.latency_ms / 10000, 1.0))
        cost_score = max(0, 1.0 - min(self.cost_usd / 1.0, 1.0))
        
        return (
            weights["accuracy"] * self.accuracy +
            weights["faithfulness"] * self.faithfulness +
            weights["hallucination"] * (1 - self.hallucination_score) +
            weights["latency"] * latency_score +
            weights["cost"] * cost_score
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt_id": self.prompt_id,
            "model_name": self.model_name,
            "provider": self.provider,
            "response": self.response,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "composite_score": self.composite_score,
        }


# =============================================================================
# Metric Base Classes
# =============================================================================


class MetricCalculator(ABC):
    """Abstract base class for metric calculators."""
    
    name: str = "base"
    description: str = ""
    
    @abstractmethod
    async def calculate(
        self,
        response: str,
        ground_truth: str | None = None,
        context: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Calculate the metric.
        
        Args:
            response: Model response
            ground_truth: Expected response
            context: Context for the prompt
            **kwargs: Additional arguments
            
        Returns:
            MetricResult object
        """
        pass


class DeterministicMetric(MetricCalculator):
    """Base class for deterministic (non-LLM) metrics."""
    pass


class LLMMetric(MetricCalculator):
    """Base class for LLM-based metrics."""
    
    def __init__(self, evaluator_provider: Any | None = None) -> None:
        """
        Initialize LLM metric.
        
        Args:
            evaluator_provider: LLM provider for evaluation
        """
        self.evaluator_provider = evaluator_provider


# =============================================================================
# Deterministic Metrics
# =============================================================================


class AccuracyMetric(DeterministicMetric):
    """
    Calculate accuracy using semantic similarity.
    Uses sentence embeddings and cosine similarity.
    """
    
    name = "accuracy"
    description = "Semantic similarity between response and ground truth"
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize accuracy metric.
        
        Args:
            embedding_model: Sentence transformer model name
        """
        self.embedding_model = embedding_model
        self._model = None
        self._lock = asyncio.Lock()
    
    async def _get_model(self) -> Any:
        """Lazy load the embedding model."""
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(self.embedding_model)
        return self._model
    
    async def calculate(
        self,
        response: str,
        ground_truth: str | None = None,
        context: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Calculate accuracy using semantic similarity.
        
        Args:
            response: Model response
            ground_truth: Expected response
            context: Context (not used for accuracy)
            
        Returns:
            MetricResult with similarity score
        """
        if ground_truth is None:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"reason": "No ground truth provided"},
            )
        
        try:
            model = await self._get_model()
            
            # Run embedding in executor to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: model.encode([response, ground_truth])
            )
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            return MetricResult(
                name=self.name,
                score=float(np.clip(similarity, 0, 1)),
                details={
                    "embedding_model": self.embedding_model,
                    "response_length": len(response),
                    "ground_truth_length": len(ground_truth),
                },
            )
            
        except Exception as e:
            logger.error(f"Accuracy calculation failed: {e}")
            return MetricResult(
                name=self.name,
                score=0.0,
                error=str(e),
            )


class ExactMatchMetric(DeterministicMetric):
    """
    Calculate exact match accuracy.
    Simple string comparison with normalization.
    """
    
    name = "exact_match"
    description = "Exact string match between response and ground truth"
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        import re
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    async def calculate(
        self,
        response: str,
        ground_truth: str | None = None,
        context: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Calculate exact match score.
        
        Args:
            response: Model response
            ground_truth: Expected response
            
        Returns:
            MetricResult with 1.0 if exact match, 0.0 otherwise
        """
        if ground_truth is None:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"reason": "No ground truth provided"},
            )
        
        norm_response = self.normalize_text(response)
        norm_ground_truth = self.normalize_text(ground_truth)
        
        match = 1.0 if norm_response == norm_ground_truth else 0.0
        
        return MetricResult(
            name=self.name,
            score=match,
            details={
                "exact_match": bool(match),
                "response_normalized": norm_response,
                "ground_truth_normalized": norm_ground_truth,
            },
        )


class ROUGEMetric(DeterministicMetric):
    """
    Calculate ROUGE-L score (Longest Common Subsequence).
    Measures word overlap between response and ground truth.
    """
    
    name = "rouge_l"
    description = "ROUGE-L F-score between response and ground truth"
    
    async def calculate(
        self,
        response: str,
        ground_truth: str | None = None,
        context: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Calculate ROUGE-L score.
        
        Args:
            response: Model response
            ground_truth: Expected response
            
        Returns:
            MetricResult with ROUGE-L score
        """
        if ground_truth is None:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"reason": "No ground truth provided"},
            )
        
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = scorer.score(ground_truth, response)
            
            rouge_l_f = scores['rougeL'].f
            
            return MetricResult(
                name=self.name,
                score=rouge_l_f,
                details={
                    "precision": scores['rougeL'].p,
                    "recall": scores['rougeL'].r,
                    "f_score": scores['rougeL'].f,
                },
            )
            
        except ImportError:
            logger.warning("rouge-score not installed, using fallback")
            return MetricResult(
                name=self.name,
                score=0.0,
                error="rouge-score not installed",
            )
        except Exception as e:
            logger.error(f"ROUGE calculation failed: {e}")
            return MetricResult(
                name=self.name,
                score=0.0,
                error=str(e),
            )


# =============================================================================
# LLM-Based Metrics
# =============================================================================


class FaithfulnessMetric(LLMMetric):
    """
    Calculate faithfulness using LLM-as-a-Judge approach.
    Evaluates if the response is supported by the provided context.
    """
    
    name = "faithfulness"
    description = "LLM-judged faithfulness to provided context"
    
    EVALUATOR_PROMPT = """You are an expert evaluator assessing whether an AI assistant's response is faithful to the provided context.

Your task: Determine if the response can be supported by the context alone, without external knowledge.

## Context:
{context}

## Response:
{response}

## Evaluation Criteria:
- 1.0: Response is fully supported by the context with no additions
- 0.7-0.9: Response is mostly supported, minor reasonable inferences
- 0.4-0.6: Response has some support but includes significant unverified claims
- 0.1-0.3: Response mostly contradicts or ignores the context
- 0.0: Response is completely unsupported or contradicts the context

## Output Format:
Return a JSON object with the following structure:
{{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<brief explanation>"
}}

Only respond with the JSON object, nothing else."""

    async def calculate(
        self,
        response: str,
        ground_truth: str | None = None,
        context: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Calculate faithfulness score using LLM judge.
        
        Args:
            response: Model response
            ground_truth: Expected response (not used)
            context: Context to evaluate against
            
        Returns:
            MetricResult with faithfulness score
        """
        if context is None:
            return MetricResult(
                name=self.name,
                score=1.0,
                details={"reason": "No context provided, assuming faithful"},
            )
        
        if self.evaluator_provider is None:
            return MetricResult(
                name=self.name,
                score=0.0,
                error="No evaluator provider configured",
            )
        
        if not response or not response.strip():
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"reason": "Empty response"},
            )
        
        try:
            prompt = self.EVALUATOR_PROMPT.format(
                context=context,
                response=response,
            )
            
            eval_response = await self.evaluator_provider.generate_response(
                prompt=prompt,
                context=None,
                model=kwargs.get("evaluator_model", "gpt-4o-mini"),
                temperature=0.1,  # Low temperature for consistent evaluation
            )
            
            if not eval_response.success:
                return MetricResult(
                    name=self.name,
                    score=0.0,
                    error=f"Evaluator failed: {eval_response.error}",
                )
            
            # Parse JSON response
            import json
            try:
                result = json.loads(eval_response.response_text)
                score = float(result.get("score", 0.0))
                reasoning = result.get("reasoning", "")
                
                return MetricResult(
                    name=self.name,
                    score=max(0.0, min(1.0, score)),  # Clamp to [0, 1]
                    details={
                        "reasoning": reasoning,
                        "evaluator_model": eval_response.model,
                    },
                )
            except json.JSONDecodeError:
                # Try to extract score from text
                text = eval_response.response_text.lower()
                if "1.0" in text or "1.00" in text:
                    return MetricResult(name=self.name, score=1.0, details={"fallback": True})
                elif "0." in text:
                    return MetricResult(name=self.name, score=0.5, details={"fallback": True})
                else:
                    return MetricResult(
                        name=self.name,
                        score=0.5,
                        error="Could not parse evaluator response",
                    )
                    
        except Exception as e:
            logger.error(f"Faithfulness calculation failed: {e}")
            return MetricResult(
                name=self.name,
                score=0.0,
                error=str(e),
            )


class HallucinationMetric(LLMMetric):
    """
    Calculate hallucination score using LLM-as-a-Judge.
    Evaluates if the response contains unsupported claims.
    """
    
    name = "hallucination"
    description = "LLM-judged hallucination (inverse of faithfulness)"
    
    EVALUATOR_PROMPT = """You are an expert fact-checker identifying hallucinations in AI responses.

Your task: Identify claims in the response that cannot be verified from the context.

## Context:
{context}

## Response:
{response}

## Evaluation:
Analyze each claim in the response and determine if it's supported by the context.
Return a JSON object:
{{
    "hallucination_score": <float 0.0 to 1.0>,
    "unsupported_claims": ["list of claims not in context"],
    "supported_claims": ["list of claims verified by context"],
    "reasoning": "<brief explanation>"
}}

A higher score means MORE hallucination (less faithful).
- 0.0: All claims are supported by context
- 0.5: Some unsupported claims
- 1.0: Most or all claims are unsupported

Only respond with the JSON object."""

    async def calculate(
        self,
        response: str,
        ground_truth: str | None = None,
        context: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Calculate hallucination score using LLM judge.
        
        Args:
            response: Model response
            ground_truth: Expected response (not used)
            context: Context to check against
            
        Returns:
            MetricResult with hallucination score
        """
        if context is None:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"reason": "No context provided, no hallucination possible"},
            )
        
        if self.evaluator_provider is None:
            return MetricResult(
                name=self.name,
                score=0.0,
                error="No evaluator provider configured",
            )
        
        if not response or not response.strip():
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"reason": "Empty response"},
            )
        
        try:
            prompt = self.EVALUATOR_PROMPT.format(
                context=context,
                response=response,
            )
            
            eval_response = await self.evaluator_provider.generate_response(
                prompt=prompt,
                context=None,
                model=kwargs.get("evaluator_model", "gpt-4o-mini"),
                temperature=0.1,
            )
            
            if not eval_response.success:
                return MetricResult(
                    name=self.name,
                    score=0.0,
                    error=f"Evaluator failed: {eval_response.error}",
                )
            
            # Parse JSON response
            import json
            try:
                result = json.loads(eval_response.response_text)
                score = float(result.get("hallucination_score", 0.0))
                
                return MetricResult(
                    name=self.name,
                    score=max(0.0, min(1.0, score)),
                    details={
                        "unsupported_claims": result.get("unsupported_claims", []),
                        "supported_claims": result.get("supported_claims", []),
                        "reasoning": result.get("reasoning", ""),
                        "evaluator_model": eval_response.model,
                    },
                )
            except json.JSONDecodeError:
                return MetricResult(
                    name=self.name,
                    score=0.5,
                    error="Could not parse evaluator response",
                )
                    
        except Exception as e:
            logger.error(f"Hallucination calculation failed: {e}")
            return MetricResult(
                name=self.name,
                score=0.0,
                error=str(e),
            )


# =============================================================================
# Metric Aggregator
# =============================================================================


class MetricAggregator:
    """
    Aggregates metrics across multiple evaluation results.
    """
    
    def __init__(self, results: list[EvaluationResult] | None = None) -> None:
        """
        Initialize aggregator.
        
        Args:
            results: Optional list of results to aggregate
        """
        self.results = results or []
    
    def add_result(self, result: EvaluationResult) -> None:
        """Add a result to the aggregator."""
        self.results.append(result)
    
    def get_model_summary(self, model_name: str) -> dict[str, Any]:
        """
        Get summary statistics for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with summary statistics
        """
        model_results = [r for r in self.results if r.model_name == model_name]
        
        if not model_results:
            return {}
        
        metrics_summary = {}
        for metric_name in ["accuracy", "faithfulness", "latency_ms", "cost_usd", "composite_score"]:
            if metric_name == "latency_ms":
                values = [r.latency_ms for r in model_results]
            elif metric_name == "cost_usd":
                values = [r.cost_usd for r in model_results]
            elif metric_name == "composite_score":
                values = [r.composite_score for r in model_results]
            else:
                values = [r.metrics.get(metric_name, MetricResult(metric_name, 0.0)).score for r in model_results]
            
            metrics_summary[metric_name] = {
                "mean": float(np.mean(values)) if values else 0.0,
                "std": float(np.std(values)) if values else 0.0,
                "min": float(np.min(values)) if values else 0.0,
                "max": float(np.max(values)) if values else 0.0,
                "count": len(values),
            }
        
        return {
            "model_name": model_name,
            "total_evaluations": len(model_results),
            "metrics": metrics_summary,
        }
    
    def get_all_summaries(self) -> dict[str, dict[str, Any]]:
        """Get summaries for all models."""
        model_names = set(r.model_name for r in self.results)
        return {name: self.get_model_summary(name) for name in model_names}
    
    def get_winner(self, metric: str = "composite_score") -> str | None:
        """
        Get the winning model for a specific metric.
        
        Args:
            metric: Metric to compare
            
        Returns:
            Name of the winning model or None
        """
        summaries = self.get_all_summaries()
        
        if not summaries:
            return None
        
        # For latency and cost, lower is better
        if metric in ["latency_ms", "cost_usd"]:
            return min(summaries, key=lambda m: summaries[m]["metrics"][metric]["mean"])
        
        # For other metrics, higher is better
        return max(summaries, key=lambda m: summaries[m]["metrics"][metric]["mean"])
    
    def to_dataframe(self) -> "pd.DataFrame":
        """Convert results to pandas DataFrame."""
        import pandas as pd
        
        data = []
        for result in self.results:
            row = {
                "prompt_id": result.prompt_id,
                "model_name": result.model_name,
                "provider": result.provider,
                "response": result.response,
                "latency_ms": result.latency_ms,
                "cost_usd": result.cost_usd,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "composite_score": result.composite_score,
            }
            
            for metric_name, metric_result in result.metrics.items():
                row[f"{metric_name}_score"] = metric_result.score
            
            data.append(row)
        
        return pd.DataFrame(data)


# Import for type hint
from pandas import DataFrame as pd_DataFrame