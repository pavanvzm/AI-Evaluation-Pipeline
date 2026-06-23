"""
Evaluation Module for AI Evaluation Pipeline.
"""
from src.evaluation.data_ingestion import (
    Dataset,
    DatasetItem,
    DatasetMetadata,
    DatasetLoader,
    DatasetValidator,
    load_dataset,
    create_sample_dataset,
)
from src.evaluation.metrics import (
    MetricResult,
    EvaluationResult,
    MetricCalculator,
    DeterministicMetric,
    LLMMetric,
    AccuracyMetric,
    ExactMatchMetric,
    ROUGEMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    MetricAggregator,
)
from src.evaluation.executor import (
    ExecutionConfig,
    ExecutionProgress,
    PromptExecutor,
    EvaluationRunner,
)

__all__ = [
    # Data Ingestion
    "Dataset",
    "DatasetItem",
    "DatasetMetadata",
    "DatasetLoader",
    "DatasetValidator",
    "load_dataset",
    "create_sample_dataset",
    # Metrics
    "MetricResult",
    "EvaluationResult",
    "MetricCalculator",
    "DeterministicMetric",
    "LLMMetric",
    "AccuracyMetric",
    "ExactMatchMetric",
    "ROUGEMetric",
    "FaithfulnessMetric",
    "HallucinationMetric",
    "MetricAggregator",
    # Executor
    "ExecutionConfig",
    "ExecutionProgress",
    "PromptExecutor",
    "EvaluationRunner",
]
