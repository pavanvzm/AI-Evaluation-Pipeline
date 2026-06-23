"""
Storage Layer for AI Evaluation Pipeline.
Handles database operations for storing and retrieving evaluation results.
"""
from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator

import aiosqlite
import pandas as pd
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from src.core.exceptions import DatabaseConnectionError, StorageError
from src.core.config import get_config
from src.evaluation.metrics import EvaluationResult

logger = logging.getLogger(__name__)

Base = declarative_base()


# =============================================================================
# SQLAlchemy Models
# =============================================================================


class EvaluationRun(Base):
    """SQLAlchemy model for evaluation runs."""
    __tablename__ = "evaluation_runs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    dataset_name = Column(String(255), nullable=False)
    dataset_path = Column(String(512))
    models_evaluated = Column(Text)  # JSON array of model names
    status = Column(String(50), default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    total_prompts = Column(Integer, default=0)
    successful_evaluations = Column(Integer, default=0)
    failed_evaluations = Column(Integer, default=0)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "dataset_name": self.dataset_name,
            "dataset_path": self.dataset_path,
            "models_evaluated": json.loads(self.models_evaluated) if self.models_evaluated else [],
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_prompts": self.total_prompts,
            "successful_evaluations": self.successful_evaluations,
            "failed_evaluations": self.failed_evaluations,
        }


class EvaluationRecord(Base):
    """SQLAlchemy model for individual evaluation records."""
    __tablename__ = "evaluation_records"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(String(36), nullable=False, index=True)
    prompt_id = Column(String(255), nullable=False)
    model_name = Column(String(255), nullable=False)
    provider = Column(String(50), nullable=False)
    prompt = Column(Text, nullable=False)
    response = Column(Text)
    ground_truth = Column(Text)
    context = Column(Text)
    
    # Token usage
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    
    # Performance metrics
    latency_ms = Column(Float, default=0.0)
    cost_usd = Column(Float, default=0.0)
    
    # Quality metrics
    accuracy_score = Column(Float, nullable=True)
    faithfulness_score = Column(Float, nullable=True)
    hallucination_score = Column(Float, nullable=True)
    composite_score = Column(Float, nullable=True)
    
    # Metadata
    metrics_details = Column(Text)  # JSON
    error = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "prompt_id": self.prompt_id,
            "model_name": self.model_name,
            "provider": self.provider,
            "prompt": self.prompt,
            "response": self.response,
            "ground_truth": self.ground_truth,
            "context": self.context,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "accuracy_score": self.accuracy_score,
            "faithfulness_score": self.faithfulness_score,
            "hallucination_score": self.hallucination_score,
            "composite_score": self.composite_score,
            "metrics_details": json.loads(self.metrics_details) if self.metrics_details else {},
            "error": self.error,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


# =============================================================================
# Database Manager
# =============================================================================


class DatabaseManager:
    """
    Manages database connections and operations for evaluation results.
    Supports both SQLite and PostgreSQL backends.
    """
    
    def __init__(self, database_url: str | None = None) -> None:
        """
        Initialize database manager.
        
        Args:
            database_url: Database connection URL. If None, uses config default.
        """
        self._config = get_config()
        self._database_url = database_url or self._get_default_database_url()
        self._engine = None
        self._async_engine = None
        self._session_factory = None
    
    def _get_default_database_url(self) -> str:
        """Get default database URL from configuration."""
        storage_config = self._config.config.storage
        db_type = storage_config.database.get("type", "sqlite")
        db_path = storage_config.database.get("path", "data/evaluation_results.db")
        
        if db_type == "sqlite":
            # Convert to async URL
            return f"sqlite+aiosqlite:///{db_path}"
        elif db_type == "postgresql":
            # Would need full PostgreSQL URL
            return storage_config.database.get("url", f"postgresql+asyncpg://localhost:5432/evaluation")
        else:
            return f"sqlite+aiosqlite:///{db_path}"
    
    async def initialize(self) -> None:
        """Initialize database and create tables."""
        try:
            # Ensure data directory exists
            db_path = self._database_url.replace("sqlite+aiosqlite:///", "")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Create async engine
            self._async_engine = create_async_engine(
                self._database_url,
                echo=False,
                future=True,
            )
            
            # Create tables
            async with self._async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info(f"Database initialized: {self._database_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DatabaseConnectionError(f"Database initialization failed: {e}")
    
    async def close(self) -> None:
        """Close database connections."""
        if self._async_engine:
            await self._async_engine.dispose()
            self._async_engine = None
            logger.info("Database connection closed")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get an async database session.
        
        Yields:
            AsyncSession instance
        """
        if self._async_engine is None:
            await self.initialize()
        
        async_session = sessionmaker(
            self._async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        
        async with async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    # =========================================================================
    # Run Operations
    # =========================================================================
    
    async def create_run(
        self,
        name: str,
        dataset_name: str,
        dataset_path: str | None = None,
        models: list[str] | None = None,
    ) -> EvaluationRun:
        """
        Create a new evaluation run.
        
        Args:
            name: Name of the run
            dataset_name: Name of the dataset
            dataset_path: Path to the dataset file
            models: List of model names being evaluated
            
        Returns:
            EvaluationRun object
        """
        async with self.get_session() as session:
            run = EvaluationRun(
                name=name,
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                models_evaluated=json.dumps(models or []),
                status="running",
            )
            session.add(run)
            await session.flush()
            return run
    
    async def update_run(
        self,
        run_id: str,
        status: str | None = None,
        total_prompts: int | None = None,
        successful: int | None = None,
        failed: int | None = None,
    ) -> None:
        """Update an evaluation run."""
        async with self.get_session() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(EvaluationRun).where(EvaluationRun.id == run_id)
            )
            run = result.scalar_one_or_none()
            
            if run:
                if status:
                    run.status = status
                if total_prompts is not None:
                    run.total_prompts = total_prompts
                if successful is not None:
                    run.successful_evaluations = successful
                if failed is not None:
                    run.failed_evaluations = failed
                if status == "completed":
                    run.completed_at = datetime.utcnow()
    
    async def get_run(self, run_id: str) -> EvaluationRun | None:
        """Get an evaluation run by ID."""
        async with self.get_session() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(EvaluationRun).where(EvaluationRun.id == run_id)
            )
            return result.scalar_one_or_none()
    
    async def get_all_runs(self) -> list[EvaluationRun]:
        """Get all evaluation runs."""
        async with self.get_session() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(EvaluationRun).order_by(EvaluationRun.created_at.desc())
            )
            return list(result.scalars().all())
    
    # =========================================================================
    # Record Operations
    # =========================================================================
    
    async def save_result(self, run_id: str, result: EvaluationResult) -> EvaluationRecord:
        """
        Save a single evaluation result.
        
        Args:
            run_id: ID of the evaluation run
            result: EvaluationResult object
            
        Returns:
            EvaluationRecord object
        """
        async with self.get_session() as session:
            # Get metric values
            accuracy = result.metrics.get("accuracy")
            faithfulness = result.metrics.get("faithfulness")
            hallucination = result.metrics.get("hallucination")
            
            record = EvaluationRecord(
                run_id=run_id,
                prompt_id=result.prompt_id,
                model_name=result.model_name,
                provider=result.provider,
                prompt="",  # Would need to store actual prompt
                response=result.response,
                ground_truth=result.ground_truth,
                context=result.context,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                total_tokens=result.input_tokens + result.output_tokens,
                latency_ms=result.latency_ms,
                cost_usd=result.cost_usd,
                accuracy_score=accuracy.score if accuracy else None,
                faithfulness_score=faithfulness.score if faithfulness else None,
                hallucination_score=hallucination.score if hallucination else None,
                composite_score=result.composite_score,
                metrics_details=json.dumps({
                    k: v.to_dict() for k, v in result.metrics.items()
                }),
                error=result.metrics.get("error").error if result.metrics.get("error") else None,
            )
            session.add(record)
            await session.flush()
            return record
    
    async def save_results(
        self,
        run_id: str,
        results: list[EvaluationResult],
    ) -> list[EvaluationRecord]:
        """
        Save multiple evaluation results.
        
        Args:
            run_id: ID of the evaluation run
            results: List of EvaluationResult objects
            
        Returns:
            List of EvaluationRecord objects
        """
        records = []
        for result in results:
            record = await self.save_result(run_id, result)
            records.append(record)
        return records
    
    async def get_results(
        self,
        run_id: str,
        model_name: str | None = None,
    ) -> list[EvaluationRecord]:
        """
        Get evaluation results for a run.
        
        Args:
            run_id: ID of the evaluation run
            model_name: Optional filter by model name
            
        Returns:
            List of EvaluationRecord objects
        """
        async with self.get_session() as session:
            from sqlalchemy import select
            query = select(EvaluationRecord).where(EvaluationRecord.run_id == run_id)
            
            if model_name:
                query = query.where(EvaluationRecord.model_name == model_name)
            
            result = await session.execute(query)
            return list(result.scalars().all())
    
    # =========================================================================
    # Aggregation & Export
    # =========================================================================
    
    async def get_results_summary(self, run_id: str) -> dict[str, Any]:
        """
        Get summary statistics for a run.
        
        Args:
            run_id: ID of the evaluation run
            
        Returns:
            Dictionary with summary statistics
        """
        async with self.get_session() as session:
            from sqlalchemy import func, select
            
            # Get all records for the run
            result = await session.execute(
                select(EvaluationRecord).where(EvaluationRecord.run_id == run_id)
            )
            records = list(result.scalars().all())
            
            if not records:
                return {"error": "No records found"}
            
            # Group by model
            model_results: dict[str, list[EvaluationRecord]] = {}
            for record in records:
                if record.model_name not in model_results:
                    model_results[record.model_name] = []
                model_results[record.model_name].append(record)
            
            # Calculate summaries
            summaries = {}
            for model_name, model_records in model_results.items():
                valid_records = [r for r in model_records if r.error is None]
                
                if valid_records:
                    summaries[model_name] = {
                        "total_evaluations": len(model_records),
                        "successful": len(valid_records),
                        "failed": len(model_records) - len(valid_records),
                        "metrics": {
                            "accuracy": {
                                "mean": self._mean([r.accuracy_score for r in valid_records if r.accuracy_score is not None]),
                                "std": self._std([r.accuracy_score for r in valid_records if r.accuracy_score is not None]),
                            },
                            "faithfulness": {
                                "mean": self._mean([r.faithfulness_score for r in valid_records if r.faithfulness_score is not None]),
                                "std": self._std([r.faithfulness_score for r in valid_records if r.faithfulness_score is not None]),
                            },
                            "hallucination": {
                                "mean": self._mean([r.hallucination_score for r in valid_records if r.hallucination_score is not None]),
                                "std": self._std([r.hallucination_score for r in valid_records if r.hallucination_score is not None]),
                            },
                            "latency_ms": {
                                "mean": self._mean([r.latency_ms for r in valid_records]),
                                "std": self._std([r.latency_ms for r in valid_records]),
                            },
                            "cost_usd": {
                                "mean": self._mean([r.cost_usd for r in valid_records]),
                                "std": self._std([r.cost_usd for r in valid_records]),
                            },
                            "composite_score": {
                                "mean": self._mean([r.composite_score for r in valid_records if r.composite_score is not None]),
                                "std": self._std([r.composite_score for r in valid_records if r.composite_score is not None]),
                            },
                        },
                    }
            
            return {
                "run_id": run_id,
                "total_records": len(records),
                "models": summaries,
            }
    
    async def export_to_dataframe(self, run_id: str) -> pd.DataFrame:
        """
        Export evaluation results to a pandas DataFrame.
        
        Args:
            run_id: ID of the evaluation run
            
        Returns:
            DataFrame with evaluation results
        """
        records = await self.get_results(run_id)
        
        data = []
        for record in records:
            data.append({
                "id": record.id,
                "run_id": record.run_id,
                "prompt_id": record.prompt_id,
                "model_name": record.model_name,
                "provider": record.provider,
                "response": record.response,
                "ground_truth": record.ground_truth,
                "context": record.context,
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "total_tokens": record.total_tokens,
                "latency_ms": record.latency_ms,
                "cost_usd": record.cost_usd,
                "accuracy_score": record.accuracy_score,
                "faithfulness_score": record.faithfulness_score,
                "hallucination_score": record.hallucination_score,
                "composite_score": record.composite_score,
                "error": record.error,
                "timestamp": record.timestamp,
            })
        
        return pd.DataFrame(data)
    
    async def export_to_csv(self, run_id: str, filepath: str | Path) -> None:
        """Export evaluation results to CSV."""
        df = await self.export_to_dataframe(run_id)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Exported results to {filepath}")
    
    async def export_to_json(self, run_id: str, filepath: str | Path) -> None:
        """Export evaluation results to JSON."""
        df = await self.export_to_dataframe(run_id)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_json(filepath, orient="records", indent=2)
        logger.info(f"Exported results to {filepath}")
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    @staticmethod
    def _mean(values: list[float]) -> float:
        """Calculate mean of a list."""
        return sum(values) / len(values) if values else 0.0
    
    @staticmethod
    def _std(values: list[float]) -> float:
        """Calculate standard deviation of a list."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5


# =============================================================================
# Singleton Database Instance
# =============================================================================


_db_manager: DatabaseManager | None = None


async def get_database() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        await _db_manager.initialize()
    return _db_manager


async def close_database() -> None:
    """Close the global database manager."""
    global _db_manager
    if _db_manager is not None:
        await _db_manager.close()
        _db_manager = None