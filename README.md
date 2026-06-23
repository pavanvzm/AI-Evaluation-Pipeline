# AI Evaluation Pipeline

A robust, production-ready system for benchmarking multiple Large Language Models (LLMs) using automated quality metrics. This pipeline reduces model validation time by 80% while delivering accurate, reliable evaluation results.

## Features

- **Multi-Model Support**: Evaluate GPT (OpenAI), Claude (Anthropic), and Llama (Groq) models
- **Comprehensive Metrics**: Accuracy, Faithfulness, Hallucination, Latency, and Cost
- **Async Execution**: Concurrent prompt processing for efficiency
- **LLM-as-a-Judge**: Sophisticated evaluation using dedicated evaluation prompts
- **Persistent Storage**: SQLite/PostgreSQL for result history
- **Export Capabilities**: CSV, JSON export for analysis tools
- **Mobile App**: PWA mobile interface for on-the-go evaluations
- **Type-Safe**: Full type hints and Pydantic models

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys

# Run evaluation
python -m src.main --dataset sample

# Run web dashboard
streamlit run src/dashboard/app.py
```

## Project Structure

```
src/
├── main.py               # CLI entry point
├── core/
│   ├── config.py         # Configuration management
│   └── exceptions.py     # Custom exceptions
├── models/
│   ├── base.py           # Abstract LLM provider
│   ├── openai_adapter.py
│   ├── anthropic_adapter.py
│   └── groq_adapter.py
├── evaluation/
│   ├── data_ingestion.py # Dataset loading
│   ├── metrics.py        # Metric calculators
│   └── executor.py       # Prompt execution
├── storage/
│   └── database.py       # SQLite/PostgreSQL storage
├── api/
│   └── production.py     # Production API server
└── dashboard/
    └── app.py            # Streamlit web dashboard

ai-eval-mobile/
└── index.html           # Mobile PWA app
```

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for Docker, Kubernetes, and production deployment guides.

## Testing

```bash
pytest
```

## License

MIT License
