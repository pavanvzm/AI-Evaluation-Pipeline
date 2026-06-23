"""
Main Orchestrator for AI Evaluation Pipeline.
Ties all components together and provides CLI interface.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.core.config import get_config, get_all_models
from src.core.exceptions import EvaluationPipelineError
from src.evaluation.data_ingestion import DatasetLoader, create_sample_dataset
from src.evaluation.executor import ExecutionConfig, EvaluationRunner
from src.evaluation.metrics import MetricAggregator
from src.models.base import LLMProviderFactory
from src.storage.database import DatabaseManager, get_database

# Import providers to register them
from src.models.openai_adapter import OpenAIProvider
from src.models.anthropic_adapter import AnthropicProvider
from src.models.groq_adapter import GroqProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CLI Interface
# =============================================================================


def setup_parser() -> argparse.ArgumentParser:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="AI Evaluation Pipeline - Benchmark LLMs on custom datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation with sample dataset
  python -m src.main --dataset sample

  # Run evaluation with custom dataset
  python -m src.main --dataset data/my_dataset.csv --models gpt-4o claude-3-5-sonnet-20240620

  # Run evaluation with specific models
  python -m src.main --dataset data/my_dataset.jsonl --models gpt-4o-mini llama-3.1-8b-instant

  # Export results
  python -m src.main --export-csv --run-id <run-id>
        """,
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset file (CSV, JSON, JSONL) or 'sample' for built-in dataset",
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        help="List of model names to evaluate (e.g., gpt-4o claude-3-5-sonnet-20240620)",
    )
    
    parser.add_argument(
        "--name",
        type=str,
        help="Name for this evaluation run",
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent API requests (default: 5)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing (default: 10)",
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for failed requests (default: 3)",
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds (default: 60)",
    )
    
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Skip quality metric calculation (faster, only latency/cost)",
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models from config",
    )
    
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export results to CSV",
    )
    
    parser.add_argument(
        "--export-json",
        action="store_true",
        help="Export results to JSON",
    )
    
    parser.add_argument(
        "--run-id",
        type=str,
        help="Run ID for export operations",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results",
        help="Output directory for exports (default: data/results)",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return parser


def list_available_models() -> None:
    """List all configured models."""
    config = get_config()
    models = config.get_all_models()
    
    print("\n" + "=" * 70)
    print("AVAILABLE MODELS")
    print("=" * 70)
    
    providers: dict[str, list[tuple[str, str]]] = {}
    for provider_name, model in models:
        if provider_name not in providers:
            providers[provider_name] = []
        providers[provider_name].append((model.name, model.display_name))
    
    for provider_name, model_list in providers.items():
        print(f"\n{provider_name.upper()}:")
        for model_name, display_name in model_list:
            print(f"  - {model_name} ({display_name})")
    
    print("\n" + "=" * 70)


# =============================================================================
# Evaluation Pipeline
# =============================================================================


class EvaluationPipeline:
    """
    Main evaluation pipeline that orchestrates all components.
    """
    
    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initialize the pipeline.
        
        Args:
            args: Command-line arguments
        """
        self.args = args
        self._config = get_config()
        self._db: DatabaseManager | None = None
        self._results: dict[str, Any] = {}
        
        # Set logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
    
    async def initialize(self) -> None:
        """Initialize database and other components."""
        logger.info("Initializing evaluation pipeline...")
        self._db = await get_database()
        logger.info("Pipeline initialized")
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._db:
            await self._db.close()
        logger.info("Pipeline cleanup complete")
    
    def _load_dataset(self) -> Any:
        """Load the dataset based on arguments."""
        if self.args.dataset == "sample":
            logger.info("Loading sample dataset...")
            return create_sample_dataset()
        
        if self.args.dataset:
            logger.info(f"Loading dataset from {self.args.dataset}...")
            return DatasetLoader.load(self.args.dataset)
        
        logger.error("No dataset specified. Use --dataset or --list-models")
        sys.exit(1)
    
    def _get_models_to_evaluate(self) -> list[tuple[str, str]]:
        """
        Get the list of models to evaluate.
        
        Returns:
            List of (provider_name, model_name) tuples
        """
        if self.args.models:
            # Map model names to providers
            models = []
            all_models = dict(get_all_models())
            
            for model_name in self.args.models:
                found = False
                for provider_name, model_config in all_models.items():
                    if model_config.name == model_name:
                        models.append((provider_name, model_name))
                        found = True
                        break
                
                if not found:
                    logger.warning(f"Model not found in config: {model_name}")
            
            if not models:
                logger.error("No valid models specified")
                sys.exit(1)
            
            return models
        
        # Default: use first model from each provider
        all_models = get_all_models()
        default_models = []
        seen_providers = set()
        
        for provider_name, model_config in all_models:
            if provider_name not in seen_providers:
                default_models.append((provider_name, model_config.name))
                seen_providers.add(provider_name)
        
        return default_models
    
    async def run(self) -> dict[str, Any]:
        """
        Run the evaluation pipeline.
        
        Returns:
            Dictionary with evaluation results
        """
        # Load dataset
        dataset = self._load_dataset()
        logger.info(f"Dataset loaded: {len(dataset)} items")
        
        # Get models to evaluate
        models = self._get_models_to_evaluate()
        logger.info(f"Evaluating {len(models)} models: {[m[1] for m in models]}")
        
        # Create execution config
        exec_config = ExecutionConfig(
            max_concurrent=self.args.max_concurrent,
            batch_size=self.args.batch_size,
            max_retries=self.args.max_retries,
            request_timeout=self.args.timeout,
        )
        
        # Create run in database
        run_name = self.args.name or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run = await self._db.create_run(
            name=run_name,
            dataset_name=dataset.metadata.name if dataset.metadata else "unknown",
            dataset_path=getattr(self.args, 'dataset', None),
            models=[m[1] for m in models],
        )
        logger.info(f"Created evaluation run: {run.id}")
        
        # Progress callback
        def progress_callback(progress: Any) -> None:
            if progress.completed % 5 == 0:
                logger.info(
                    f"Progress: {progress.completed}/{progress.total} "
                    f"({progress.percentage:.1f}%) - "
                    f"Model: {progress.current_model}"
                )
        
        # Create evaluator provider for LLM-based metrics
        evaluator_provider = None
        if not self.args.skip_metrics:
            try:
                from src.models.openai_adapter import OpenAIProvider
                evaluator_provider = OpenAIProvider()
                logger.info("Evaluator provider initialized")
            except Exception as e:
                logger.warning(f"Could not initialize evaluator provider: {e}")
        
        # Create evaluation runner
        runner = EvaluationRunner(
            evaluator_provider=evaluator_provider,
            execution_config=exec_config,
        )
        
        # Run evaluation
        try:
            logger.info("Starting evaluation...")
            results = await runner.run(
                dataset=dataset,
                models=models,
                calculate_metrics=not self.args.skip_metrics,
            )
            
            # Save results to database
            total_evaluations = 0
            successful = 0
            
            for model_name, model_results in results.items():
                await self._db.save_results(run.id, model_results)
                total_evaluations += len(model_results)
                successful += sum(1 for r in model_results if not r.metrics.get("error"))
            
            # Update run status
            await self._db.update_run(
                run.id,
                status="completed",
                total_prompts=total_evaluations,
                successful=successful,
                failed=total_evaluations - successful,
            )
            
            logger.info(f"Evaluation completed: {successful}/{total_evaluations} successful")
            
            self._results = {
                "run_id": run.id,
                "run_name": run_name,
                "dataset": dataset.name,
                "models": models,
                "results": results,
            }
            
            return self._results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            await self._db.update_run(run.id, status="failed")
            raise
    
    async def export_results(self) -> None:
        """Export results to files."""
        if not self.args.run_id:
            logger.error("No run-id specified for export")
            return
        
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.args.export_csv:
            filepath = output_dir / f"{self.args.run_id}.csv"
            await self._db.export_to_csv(self.args.run_id, filepath)
            print(f"Exported to: {filepath}")
        
        if self.args.export_json:
            filepath = output_dir / f"{self.args.run_id}.json"
            await self._db.export_to_json(self.args.run_id, filepath)
            print(f"Exported to: {filepath}")
    
    def print_summary(self) -> None:
        """Print a summary of the evaluation results."""
        if not self._results:
            return
        
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Run ID: {self._results['run_id']}")
        print(f"Dataset: {self._results['dataset']}")
        print(f"Models: {[m[1] for m in self._results['models']]}")
        
        # Calculate summary statistics
        aggregator = MetricAggregator()
        for model_results in self._results['results'].values():
            for result in model_results:
                aggregator.add_result(result)
        
        summaries = aggregator.get_all_summaries()
        
        print("\n" + "-" * 70)
        print(f"{'Model':<30} {'Accuracy':<12} {'Faithful':<12} {'Latency':<12} {'Cost':<12}")
        print("-" * 70)
        
        for model_name, summary in summaries.items():
            metrics = summary["metrics"]
            print(
                f"{model_name:<30} "
                f"{metrics['accuracy']['mean']:.3f}        "
                f"{metrics['faithfulness']['mean']:.3f}        "
                f"{metrics['latency_ms']['mean']:.1f}ms      "
                f"${metrics['cost_usd']['mean']:.4f}     "
            )
        
        print("-" * 70)
        
        # Determine winner
        winner = aggregator.get_winner("composite_score")
        if winner:
            print(f"\nOverall Winner: {winner}")
        
        print("=" * 70)


# =============================================================================
# Main Entry Point
# =============================================================================


async def main_async(args: argparse.Namespace) -> None:
    """Async main function."""
    try:
        pipeline = EvaluationPipeline(args)
        await pipeline.initialize()
        
        try:
            if args.list_models:
                list_available_models()
            elif args.export_csv or args.export_json:
                await pipeline.export_results()
            else:
                await pipeline.run()
                pipeline.print_summary()
        finally:
            await pipeline.cleanup()
            
    except EvaluationPipelineError as e:
        logger.error(f"Pipeline error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()