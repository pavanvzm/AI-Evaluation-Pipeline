# AI Evaluation Pipeline

A robust, production-ready system for benchmarking multiple Large Language Models (LLMs) using automated quality metrics. This pipeline reduces model validation time by 80% while delivering accurate, reliable evaluation results.

## Features

- **Multi-Model Support**: Evaluate GPT (OpenAI), Claude (Anthropic), and Llama (Groq) models
- **Comprehensive Metrics**: Accuracy, Faithfulness, Hallucination, Latency, and Cost
- **Async Execution**: Concurrent prompt processing for efficiency
- **LLM-as-a-Judge**: Sophisticated evaluation using dedicated evaluation prompts
- **Persistent Storage**: SQLite/PostgreSQL for result history
- **Export Capabilities**: CSV, JSON export for analysis tools
- **Type-Safe**: Full type hints and Pydantic models

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI / API Layer                           │
│  - FastAPI REST Endpoints (optional)                             │
│  - Streamlit Dashboard (optional)                                │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                     Orchestration Layer                          │
│  - Prompt Executor                                               │
│  - Evaluation Runner                                             │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                     Evaluation Engine                            │
│  - Metric Calculators (Accuracy, Faithfulness, etc.)            │
│  - LLM-as-a-Judge Integration                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                     Model Adapters                               │
│  - OpenAI Adapter (GPT-4o, GPT-4o-mini, GPT-4-Turbo)            │
│  - Anthropic Adapter (Claude 3.5 Sonnet, Claude 3 Opus, etc.)   │
│  - Groq Adapter (Llama 3.1 70B, Llama 3.1 8B)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                     Data Layer                                   │
│  - Dataset Loader (CSV, JSON, JSONL)                             │
│  - SQLite/PostgreSQL Storage                                     │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- API keys for the LLM providers you want to use

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd AI-Evaluation-Pipeline
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure API keys:
```bash
cp .env.example .env
# Edit .env and add your API keys:
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# GROQ_API_KEY=gsk_...
```

5. Verify configuration:
```bash
python -m src.main --list-models
```

## Usage

### Quick Start with Sample Dataset

```bash
python -m src.main --dataset sample
```

### Evaluate with Custom Dataset

```bash
# Using CSV
python -m src.main --dataset data/my_dataset.csv --models gpt-4o claude-3-5-sonnet-20240620

# Using JSONL
python -m src.main --dataset data/my_dataset.jsonl --models gpt-4o-mini llama-3.1-8b-instant
```

### Available Command-Line Options

```bash
python -m src.main --help

Options:
  --dataset PATH              Path to dataset file (CSV, JSON, JSONL) or 'sample'
  --models MODEL [MODEL ...]  List of model names to evaluate
  --name TEXT                 Name for this evaluation run
  --max-concurrent INT        Max concurrent API requests (default: 5)
  --batch-size INT            Batch size for processing (default: 10)
  --max-retries INT           Max retry attempts (default: 3)
  --timeout INT               Request timeout in seconds (default: 60)
  --skip-metrics              Skip quality metrics (faster, only latency/cost)
  --list-models               List all available models
  --export-csv                Export results to CSV
  --export-json               Export results to JSON
  --run-id TEXT               Run ID for export operations
  --output-dir PATH           Output directory (default: data/results)
  --verbose                   Enable verbose logging
```

### Programmatic Usage

```python
import asyncio
from src.evaluation.data_ingestion import create_sample_dataset
from src.evaluation.executor import ExecutionConfig, EvaluationRunner
from src.models.openai_adapter import OpenAIProvider
from src.storage.database import get_database

async def main():
    # Initialize database
    db = await get_database()
    
    # Load dataset
    dataset = create_sample_dataset()
    
    # Configure execution
    config = ExecutionConfig(max_concurrent=5)
    
    # Create evaluator provider
    evaluator = OpenAIProvider()
    
    # Create runner
    runner = EvaluationRunner(
        evaluator_provider=evaluator,
        execution_config=config,
    )
    
    # Define models to evaluate
    models = [
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-3-5-sonnet-20240620"),
        ("groq", "llama-3.1-8b-instant"),
    ]
    
    # Run evaluation
    results = await runner.run(
        dataset=dataset,
        models=models,
        calculate_metrics=True,
    )
    
    # Print summary
    for model_name, model_results in results.items():
        avg_accuracy = sum(r.accuracy for r in model_results) / len(model_results)
        avg_latency = sum(r.latency_ms for r in model_results) / len(model_results)
        print(f"{model_name}: Accuracy={avg_accuracy:.3f}, Latency={avg_latency:.1f}ms")
    
    await db.close()

asyncio.run(main())
```

## Dataset Format

### CSV Format

```csv
id,prompt,ground_truth,context
q1,What is your return policy?,You can return items within 30 days...,Our return policy allows...
q2,How do I reset my password?,Click 'Forgot Password'...,To reset your password...
```

### JSON Format

```json
[
  {
    "id": "q1",
    "prompt": "What is your return policy?",
    "ground_truth": "You can return items within 30 days...",
    "context": "Our return policy allows..."
  }
]
```

### JSONL Format

```jsonl
{"id": "q1", "prompt": "What is your return policy?", "ground_truth": "You can return...", "context": "..."}
{"id": "q2", "prompt": "How do I reset my password?", "ground_truth": "Click...", "context": "..."}
```

## Metrics

### Quality Metrics

| Metric | Description | Method |
|--------|-------------|--------|
| **Accuracy** | Semantic similarity to ground truth | Sentence embeddings + cosine similarity |
| **Faithfulness** | Alignment with provided context | LLM-as-a-Judge evaluation |
| **Hallucination** | Unsupported claims in response | LLM-as-a-Judge evaluation |

### Performance Metrics

| Metric | Description |
|--------|-------------|
| **Latency** | Time from request to response (ms) |
| **Cost** | API cost per evaluation (USD) |
| **Tokens** | Input + output token count |

### Composite Score

A weighted combination of all metrics:
- Accuracy: 30%
- Faithfulness: 25%
- Hallucination: 25%
- Latency: 10%
- Cost: 10%

## Configuration

### config/config.yaml

Configure model providers, pricing, and evaluation settings:

```yaml
models:
  openai:
    provider: "openai"
    models:
      - name: "gpt-4o"
        display_name: "GPT-4o"
        pricing:
          input_cost_per_1k: 0.005
          output_cost_per_1k: 0.015

evaluation:
  max_concurrent_requests: 5
  max_retries: 3
  request_timeout: 60
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | For OpenAI models |
| `ANTHROPIC_API_KEY` | Anthropic API key | For Claude models |
| `GROQ_API_KEY` | Groq API key | For Llama models |
| `MAX_CONCURRENT_REQUESTS` | Max concurrent requests | No |
| `REQUEST_TIMEOUT` | Request timeout (seconds) | No |
| `LOG_LEVEL` | Logging level | No |

## Project Structure

```
ai-evaluation-pipeline/
├── config/
│   ├── config.yaml           # Model configurations
│   └── prompts.py            # Evaluation prompt templates
├── src/
│   ├── __init__.py
│   ├── main.py               # CLI entry point
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py         # Configuration management
│   │   └── exceptions.py     # Custom exceptions
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py           # Abstract LLM provider
│   │   ├── openai_adapter.py
│   │   ├── anthropic_adapter.py
│   │   └── groq_adapter.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py # Dataset loading
│   │   ├── metrics.py        # Metric calculators
│   │   └── executor.py       # Prompt execution
│   └── storage/
│       ├── __init__.py
│       └── database.py       # SQLite/PostgreSQL storage
├── data/
│   ├── datasets/             # Benchmark datasets
│   └── results/              # Evaluation results
├── tests/                    # Unit tests
├── .env.example              # Environment template
├── requirements.txt
└── README.md
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_metrics.py -v
```

## Error Handling

The pipeline implements robust error handling:

- **Retry Logic**: Exponential backoff for API failures
- **Rate Limiting**: Automatic handling of rate limit errors
- **Timeout Protection**: Configurable timeouts prevent hanging
- **Partial Results**: Saves successful evaluations even if some fail
- **Graceful Degradation**: Continues with available models if one fails

## Troubleshooting

### Common Issues

**API Key Not Found**
```
Error: OPENAI_API_KEY environment variable not set
```
Solution: Add your API key to the `.env` file.

**Rate Limit Exceeded**
```
Warning: Rate limit hit for gpt-4o
```
Solution: Reduce `--max-concurrent` or wait and retry.

**Dataset Validation Error**
```
Error: Missing required columns: ['prompt']
```
Solution: Ensure your dataset has the required columns (`id`, `prompt`).

### Debug Mode

Enable verbose logging:
```bash
python -m src.main --dataset sample --verbose
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## Support

For issues or questions, please open a GitHub issue.