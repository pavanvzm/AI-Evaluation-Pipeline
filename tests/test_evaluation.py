"""
Unit tests for AI Evaluation Pipeline.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.evaluation.data_ingestion import (
    Dataset,
    DatasetItem,
    DatasetLoader,
    DatasetValidator,
    create_sample_dataset,
)
from src.evaluation.metrics import (
    MetricResult,
    EvaluationResult,
    AccuracyMetric,
    ExactMatchMetric,
    MetricAggregator,
)
from src.models.base import LLMResponse, LLMProviderFactory
from src.core.config import ConfigManager, Settings


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return create_sample_dataset()


@pytest.fixture
def sample_items():
    """Create sample dataset items."""
    return [
        DatasetItem(
            id="test1",
            prompt="What is 2+2?",
            ground_truth="4",
            context="Basic math.",
        ),
        DatasetItem(
            id="test2",
            prompt="What is the capital of France?",
            ground_truth="Paris",
            context="European geography.",
        ),
    ]


@pytest.fixture
def sample_response():
    """Create a sample LLM response."""
    return LLMResponse(
        model="gpt-4o-mini",
        response_text="4",
        input_tokens=10,
        output_tokens=2,
        latency_ms=500.0,
        cost_usd=0.0001,
    )


@pytest.fixture
def sample_evaluation_result():
    """Create a sample evaluation result."""
    return EvaluationResult(
        prompt_id="test1",
        model_name="gpt-4o-mini",
        provider="openai",
        response="4",
        latency_ms=500.0,
        cost_usd=0.0001,
        input_tokens=10,
        output_tokens=2,
        ground_truth="4",
        context="Basic math.",
        metrics={
            "accuracy": MetricResult("accuracy", 0.95),
            "faithfulness": MetricResult("faithfulness", 0.90),
        },
    )


# =============================================================================
# Data Ingestion Tests
# =============================================================================


class TestDatasetItem:
    """Tests for DatasetItem model."""
    
    def test_create_valid_item(self):
        """Test creating a valid dataset item."""
        item = DatasetItem(
            id="test1",
            prompt="What is 2+2?",
            ground_truth="4",
            context="Basic math.",
        )
        
        assert item.id == "test1"
        assert item.prompt == "What is 2+2?"
        assert item.ground_truth == "4"
        assert item.context == "Basic math."
    
    def test_prompt_validation(self):
        """Test prompt validation rejects empty prompts."""
        with pytest.raises(ValueError):
            DatasetItem(id="test1", prompt="")
        
        with pytest.raises(ValueError):
            DatasetItem(id="test1", prompt="   ")
    
    def test_prompt_trimming(self):
        """Test prompt whitespace trimming."""
        item = DatasetItem(id="test1", prompt="  Hello  ")
        assert item.prompt == "Hello"
    
    def test_optional_fields(self):
        """Test optional fields default to None."""
        item = DatasetItem(id="test1", prompt="Question?")
        assert item.ground_truth is None
        assert item.context is None
        assert item.metadata == {}


class TestDataset:
    """Tests for Dataset class."""
    
    def test_create_dataset(self, sample_items):
        """Test creating a dataset."""
        dataset = Dataset(name="test", items=sample_items)
        
        assert len(dataset) == 2
        assert dataset.name == "test"
        assert dataset[0].id == "test1"
    
    def test_dataset_iteration(self, sample_items):
        """Test dataset iteration."""
        dataset = Dataset(name="test", items=sample_items)
        
        ids = [item.id for item in dataset]
        assert ids == ["test1", "test2"]
    
    def test_get_items_with_ground_truth(self, sample_items):
        """Test filtering items with ground truth."""
        dataset = Dataset(name="test", items=sample_items)
        
        with_gt = dataset.get_items_with_ground_truth()
        assert len(with_gt) == 2
    
    def test_get_items_with_context(self, sample_items):
        """Test filtering items with context."""
        dataset = Dataset(name="test", items=sample_items)
        
        with_context = dataset.get_items_with_context()
        assert len(with_context) == 2
    
    def test_to_dataframe(self, sample_items):
        """Test converting dataset to DataFrame."""
        dataset = Dataset(name="test", items=sample_items)
        df = dataset.to_dataframe()
        
        assert len(df) == 2
        assert "prompt" in df.columns
        assert "id" in df.columns


class TestDatasetValidator:
    """Tests for DatasetValidator."""
    
    def test_validate_file_not_found(self):
        """Test validation fails for missing file."""
        is_valid, error = DatasetValidator.validate_file("nonexistent.csv")
        assert not is_valid
        assert "not found" in error.lower()
    
    def test_validate_file_unsupported_format(self, tmp_path):
        """Test validation fails for unsupported format."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("content")
        
        is_valid, error = DatasetValidator.validate_file(file_path)
        assert not is_valid
        assert "unsupported" in error.lower()
    
    def test_validate_dataframe_missing_columns(self):
        """Test validation fails for missing columns."""
        import pandas as pd
        
        df = pd.DataFrame({"prompt": ["test"]})
        is_valid, error, missing = DatasetValidator.validate_dataframe(df)
        
        assert not is_valid
        assert "id" in missing
    
    def test_validate_dataframe_empty(self):
        """Test validation fails for empty DataFrame."""
        import pandas as pd
        
        df = pd.DataFrame(columns=["id", "prompt"])
        is_valid, error, _ = DatasetValidator.validate_dataframe(df)
        
        assert not is_valid
        assert "empty" in error.lower()


class TestDatasetLoader:
    """Tests for DatasetLoader."""
    
    def test_load_jsonl(self, tmp_path):
        """Test loading JSONL file."""
        file_path = tmp_path / "test.jsonl"
        file_path.write_text(
            '{"id": "q1", "prompt": "What is 2+2?", "ground_truth": "4"}\n'
            '{"id": "q2", "prompt": "What is 3+3?", "ground_truth": "6"}\n'
        )
        
        dataset = DatasetLoader.load(file_path, "test")
        
        assert len(dataset) == 2
        assert dataset.name == "test"
        assert dataset[0].prompt == "What is 2+2?"
    
    def test_load_json(self, tmp_path):
        """Test loading JSON file."""
        file_path = tmp_path / "test.json"
        file_path.write_text(
            '[{"id": "q1", "prompt": "What is 2+2?", "ground_truth": "4"}]'
        )
        
        dataset = DatasetLoader.load(file_path, "test")
        
        assert len(dataset) == 1
        assert dataset[0].ground_truth == "4"
    
    def test_load_csv(self, tmp_path):
        """Test loading CSV file."""
        file_path = tmp_path / "test.csv"
        file_path.write_text("id,prompt,ground_truth\nq1,What is 2+2?,4")
        
        dataset = DatasetLoader.load(file_path, "test")
        
        assert len(dataset) == 1
        assert dataset[0].id == "q1"


class TestSampleDataset:
    """Tests for sample dataset creation."""
    
    def test_create_sample_dataset(self):
        """Test creating sample dataset."""
        dataset = create_sample_dataset()
        
        assert len(dataset) == 5
        assert dataset.name == "sample_customer_support"
        
        # Check all items have required fields
        for item in dataset:
            assert item.id
            assert item.prompt
            assert item.ground_truth
            assert item.context
    
    def test_sample_dataset_summary(self):
        """Test sample dataset summary."""
        dataset = create_sample_dataset()
        summary = dataset.summary()
        
        assert summary["total_items"] == 5
        assert summary["items_with_ground_truth"] == 5
        assert summary["items_with_context"] == 5


# =============================================================================
# Metrics Tests
# =============================================================================


class TestMetricResult:
    """Tests for MetricResult."""
    
    def test_create_metric_result(self):
        """Test creating a metric result."""
        result = MetricResult(name="accuracy", score=0.95)
        
        assert result.name == "accuracy"
        assert result.score == 0.95
        assert result.success
        assert result.error is None
    
    def test_metric_result_with_error(self):
        """Test metric result with error."""
        result = MetricResult(name="accuracy", score=0.0, error="Test error")
        
        assert not result.success
        assert result.error == "Test error"
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        result = MetricResult(
            name="accuracy",
            score=0.95,
            details={"model": "test"},
        )
        
        d = result.to_dict()
        assert d["name"] == "accuracy"
        assert d["score"] == 0.95
        assert d["details"]["model"] == "test"


class TestEvaluationResult:
    """Tests for EvaluationResult."""
    
    def test_create_evaluation_result(self, sample_response):
        """Test creating evaluation result."""
        result = EvaluationResult(
            prompt_id="test1",
            model_name="gpt-4o-mini",
            provider="openai",
            response=sample_response.response_text,
            latency_ms=sample_response.latency_ms,
            cost_usd=sample_response.cost_usd,
            input_tokens=sample_response.input_tokens,
            output_tokens=sample_response.output_tokens,
        )
        
        assert result.prompt_id == "test1"
        assert result.response == "4"
        assert result.composite_score >= 0
    
    def test_composite_score_calculation(self, sample_evaluation_result):
        """Test composite score calculation."""
        score = sample_evaluation_result.composite_score
        
        assert 0 <= score <= 1
        # Should be high since accuracy and faithfulness are high
        assert score > 0.5


class TestExactMatchMetric:
    """Tests for ExactMatchMetric."""
    
    @pytest.mark.asyncio
    async def test_exact_match_same(self):
        """Test exact match with identical strings."""
        metric = ExactMatchMetric()
        
        result = await metric.calculate(
            response="The answer is four",
            ground_truth="the answer is four",
        )
        
        assert result.score == 1.0
    
    @pytest.mark.asyncio
    async def test_exact_match_different(self):
        """Test exact match with different strings."""
        metric = ExactMatchMetric()
        
        result = await metric.calculate(
            response="The answer is five",
            ground_truth="the answer is four",
        )
        
        assert result.score == 0.0
    
    @pytest.mark.asyncio
    async def test_exact_match_no_ground_truth(self):
        """Test with no ground truth."""
        metric = ExactMatchMetric()
        
        result = await metric.calculate(response="Test")
        
        assert result.score == 0.0
        assert "No ground truth" in result.details["reason"]


class TestMetricAggregator:
    """Tests for MetricAggregator."""
    
    def test_add_result(self, sample_evaluation_result):
        """Test adding results to aggregator."""
        aggregator = MetricAggregator()
        aggregator.add_result(sample_evaluation_result)
        
        assert len(aggregator.results) == 1
    
    def test_get_model_summary(self, sample_evaluation_result):
        """Test getting model summary."""
        aggregator = MetricAggregator()
        aggregator.add_result(sample_evaluation_result)
        
        summary = aggregator.get_model_summary("gpt-4o-mini")
        
        assert summary["total_evaluations"] == 1
        assert "metrics" in summary
    
    def test_get_winner(self, sample_evaluation_result):
        """Test getting winning model."""
        # Create another result with lower score
        result2 = EvaluationResult(
            prompt_id="test2",
            model_name="claude-3-5-sonnet",
            provider="anthropic",
            response="Paris",
            latency_ms=800.0,
            cost_usd=0.0002,
            input_tokens=15,
            output_tokens=3,
            metrics={
                "accuracy": MetricResult("accuracy", 0.80),
                "faithfulness": MetricResult("faithfulness", 0.75),
            },
        )
        
        aggregator = MetricAggregator()
        aggregator.add_result(sample_evaluation_result)
        aggregator.add_result(result2)
        
        winner = aggregator.get_winner("accuracy")
        assert winner == "gpt-4o-mini"  # Has higher accuracy
    
    def test_to_dataframe(self, sample_evaluation_result):
        """Test converting to DataFrame."""
        aggregator = MetricAggregator()
        aggregator.add_result(sample_evaluation_result)
        
        df = aggregator.to_dataframe()
        
        assert len(df) == 1
        assert "model_name" in df.columns
        assert "composite_score" in df.columns


# =============================================================================
# Provider Tests
# =============================================================================


class TestLLMProviderFactory:
    """Tests for LLMProviderFactory."""
    
    def test_register_provider(self):
        """Test registering a provider."""
        class TestProvider:
            provider_name = "test"
            
            async def generate_response(self, prompt, context=None):
                return MagicMock()
        
        LLMProviderFactory.register("test", TestProvider)
        
        assert "test" in LLMProviderFactory.get_registered_providers()
    
    def test_create_provider(self):
        """Test creating a provider."""
        # Register a mock provider
        class MockProvider:
            provider_name = "mock"
            
            def __init__(self, **kwargs):
                pass
            
            async def generate_response(self, prompt, context=None):
                return MagicMock()
        
        LLMProviderFactory.register("mock", MockProvider)
        
        provider = LLMProviderFactory.create("mock")
        assert provider is not None
    
    def test_create_unknown_provider(self):
        """Test creating unknown provider raises error."""
        with pytest.raises(ValueError) as exc_info:
            LLMProviderFactory.create("nonexistent")
        
        assert "Unknown provider" in str(exc_info.value)


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfigManager:
    """Tests for ConfigManager."""
    
    def test_singleton(self):
        """Test ConfigManager is singleton."""
        config1 = ConfigManager()
        config2 = ConfigManager()
        
        assert config1 is config2
    
    def test_get_api_key(self):
        """Test getting API key."""
        config = ConfigManager()
        
        # Should not raise even with missing key
        key = config.get_api_key("openai")
        # Key may be None if not set in environment
    
    def test_calculate_cost(self):
        """Test cost calculation."""
        config = ConfigManager()
        
        # Using default config values
        cost = config.calculate_cost("gpt-4o", 1000, 500)
        
        # Cost should be positive for valid models
        # May be 0 if model not configured
        assert cost >= 0


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])