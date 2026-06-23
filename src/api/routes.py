"""
FastAPI REST API for AI Evaluation Pipeline.
Provides endpoints for managing evaluations, datasets, and results.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from src.core.config import get_config, get_all_models
from src.core.exceptions import EvaluationPipelineError, DatasetError
from src.evaluation.data_ingestion import DatasetLoader, create_sample_dataset, Dataset, DatasetItem
from src.evaluation.executor import ExecutionConfig, EvaluationRunner
from src.evaluation.metrics import MetricAggregator
from src.storage.database import DatabaseManager, get_database, close_database

logger = logging.getLogger(__name__)

# =============================================================================
# Pydantic Models
# =============================================================================


class EvaluationRequest(BaseModel):
    """Request model for starting an evaluation."""
    dataset_name: str = Field(..., description="Name of the dataset")
    models: list[str] = Field(..., description="List of model names to evaluate")
    calculate_metrics: bool = Field(default=True, description="Calculate quality metrics")
    max_concurrent: int = Field(default=5, ge=1, le=20, description="Max concurrent requests")
    batch_size: int = Field(default=10, ge=1, le=100, description="Batch size")
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retry attempts")
    timeout: int = Field(default=60, ge=10, le=300, description="Request timeout in seconds")


class EvaluationResponse(BaseModel):
    """Response model for evaluation status."""
    run_id: str
    status: str
    message: str


class RunStatus(BaseModel):
    """Model for run status information."""
    id: str
    name: str
    dataset_name: str
    status: str
    total_prompts: int
    successful_evaluations: int
    failed_evaluations: int
    created_at: str | None
    completed_at: str | None


class ModelInfo(BaseModel):
    """Model for model information."""
    name: str
    display_name: str
    provider: str


class HealthStatus(BaseModel):
    """Model for health check response."""
    status: str
    version: str
    timestamp: str
    services: dict[str, str]


# =============================================================================
# FastAPI Application
# =============================================================================


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="AI Evaluation Pipeline API",
        description="REST API for benchmarking and evaluating LLMs",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Store for background tasks
    app.state.active_runs: dict[str, asyncio.Task] = {}
    
    return app


app = create_app()


# =============================================================================
# Health & Info Endpoints
# =============================================================================


@app.get("/health", response_model=HealthStatus, tags=["Health"])
async def health_check() -> HealthStatus:
    """
    Health check endpoint.
    Returns the status of the service and its dependencies.
    """
    services = {"database": "unknown", "config": "ok"}
    
    # Check database
    try:
        db = await get_database()
        await db.get_all_runs()
        services["database"] = "ok"
    except Exception as e:
        services["database"] = f"error: {str(e)}"
    
    return HealthStatus(
        status="healthy" if services["database"] == "ok" else "degraded",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat(),
        services=services,
    )


@app.get("/info", tags=["Info"])
async def get_info() -> dict[str, Any]:
    """
    Get service information and available models.
    """
    config = get_config()
    all_models = get_all_models()
    
    models = []
    for provider_name, model_config in all_models:
        models.append(ModelInfo(
            name=model_config.name,
            display_name=model_config.display_name,
            provider=provider_name,
        ))
    
    return {
        "service": "AI Evaluation Pipeline",
        "version": "1.0.0",
        "available_models": [m.model_dump() for m in models],
        "supported_formats": ["csv", "json", "jsonl"],
    }


# =============================================================================
# Dataset Endpoints
# =============================================================================


@app.get("/datasets", tags=["Datasets"])
async def list_datasets() -> list[dict[str, Any]]:
    """
    List available datasets in the data directory.
    """
    datasets_dir = Path("data/datasets")
    if not datasets_dir.exists():
        return []
    
    datasets = []
    for file_path in datasets_dir.glob("*"):
        if file_path.suffix.lower() in [".csv", ".json", ".jsonl"]:
            try:
                dataset = DatasetLoader.load(file_path)
                datasets.append({
                    "name": dataset.name,
                    "path": str(file_path),
                    "file_size": file_path.stat().st_size,
                    "item_count": len(dataset),
                    "has_ground_truth": dataset.metadata.has_ground_truth if dataset.metadata else False,
                    "has_context": dataset.metadata.has_context if dataset.metadata else False,
                })
            except Exception as e:
                logger.warning(f"Failed to load dataset {file_path}: {e}")
    
    return datasets


@app.post("/datasets/upload", tags=["Datasets"])
async def upload_dataset(
    file: UploadFile = File(...),
    name: str | None = None,
) -> dict[str, Any]:
    """
    Upload a dataset file.
    Supports CSV, JSON, and JSONL formats.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Validate file type
    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".csv", ".json", ".jsonl"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {suffix}. Supported: .csv, .json, .jsonl"
        )
    
    # Save file
    datasets_dir = Path("data/datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_name = name or Path(file.filename).stem
    file_path = datasets_dir / f"{dataset_name}{suffix}"
    
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Validate dataset
    try:
        dataset = DatasetLoader.load(file_path, dataset_name)
        return {
            "message": "Dataset uploaded successfully",
            "name": dataset.name,
            "path": str(file_path),
            "item_count": len(dataset),
        }
    except (DatasetError, ValueError) as e:
        # Clean up invalid file
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/datasets/{dataset_name}", tags=["Datasets"])
async def get_dataset(dataset_name: str) -> dict[str, Any]:
    """
    Get dataset details and preview items.
    """
    datasets_dir = Path("data/datasets")
    
    # Search for dataset file
    for suffix in [".csv", ".json", ".jsonl"]:
        file_path = datasets_dir / f"{dataset_name}{suffix}"
        if file_path.exists():
            break
    else:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_name}")
    
    try:
        dataset = DatasetLoader.load(file_path, dataset_name)
        summary = dataset.summary()
        
        # Get first 5 items as preview
        preview = []
        for item in dataset[:5]:
            preview.append({
                "id": item.id,
                "prompt": item.prompt[:100] + "..." if len(item.prompt) > 100 else item.prompt,
                "has_ground_truth": item.ground_truth is not None,
                "has_context": item.context is not None,
            })
        
        return {
            **summary,
            "preview": preview,
        }
    except DatasetError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
# Evaluation Endpoints
# =============================================================================


@app.post("/evaluations", response_model=EvaluationResponse, tags=["Evaluations"])
async def create_evaluation(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
) -> EvaluationResponse:
    """
    Start a new evaluation run.
    Returns immediately with a run_id for tracking.
    """
    # Validate dataset exists
    datasets_dir = Path("data/datasets")
    dataset_path = None
    
    for suffix in [".csv", ".json", ".jsonl"]:
        path = datasets_dir / f"{request.dataset_name}{suffix}"
        if path.exists():
            dataset_path = path
            break
    
    if dataset_path is None:
        if request.dataset_name == "sample":
            dataset_path = "sample"
        else:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_name}")
    
    # Validate models
    all_models = dict(get_all_models())
    valid_models = []
    
    for model_name in request.models:
        found = False
        for provider_name, model_config in all_models.items():
            if model_config.name == model_name:
                valid_models.append((provider_name, model_name))
                found = True
                break
        if not found:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    
    if not valid_models:
        raise HTTPException(status_code=400, detail="No valid models specified")
    
    # Create run
    db = await get_database()
    run = await db.create_run(
        name=f"api_run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        dataset_name=request.dataset_name,
        dataset_path=str(dataset_path) if dataset_path != "sample" else None,
        models=request.models,
    )
    run_id = run.id
    
    # Start background evaluation
    async def run_evaluation():
        try:
            # Load dataset
            if dataset_path == "sample":
                dataset = create_sample_dataset()
            else:
                dataset = DatasetLoader.load(dataset_path, request.dataset_name)
            
            # Configure execution
            exec_config = ExecutionConfig(
                max_concurrent=request.max_concurrent,
                batch_size=request.batch_size,
                max_retries=request.max_retries,
                request_timeout=request.timeout,
            )
            
            # Create evaluator provider
            evaluator = None
            if request.calculate_metrics:
                try:
                    from src.models.openai_adapter import OpenAIProvider
                    evaluator = OpenAIProvider()
                except Exception as e:
                    logger.warning(f"Could not create evaluator: {e}")
            
            # Run evaluation
            runner = EvaluationRunner(
                evaluator_provider=evaluator,
                execution_config=exec_config,
            )
            
            results = await runner.run(
                dataset=dataset,
                models=valid_models,
                calculate_metrics=request.calculate_metrics,
            )
            
            # Save results
            for model_name, model_results in results.items():
                await db.save_results(run_id, model_results)
            
            # Update run status
            total = sum(len(r) for r in results.values())
            successful = sum(
                sum(1 for r in model_results if not r.metrics.get("error"))
                for model_results in results.values()
            )
            await db.update_run(
                run_id,
                status="completed",
                total_prompts=total,
                successful=successful,
                failed=total - successful,
            )
            
            logger.info(f"Evaluation {run_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Evaluation {run_id} failed: {e}")
            await db.update_run(run_id, status="failed")
        finally:
            if run_id in app.state.active_runs:
                del app.state.active_runs[run_id]
    
    task = asyncio.create_task(run_evaluation())
    app.state.active_runs[run_id] = task
    
    return EvaluationResponse(
        run_id=run_id,
        status="started",
        message=f"Evaluation started for {len(valid_models)} models",
    )


@app.get("/evaluations", tags=["Evaluations"])
async def list_evaluations(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """
    List all evaluation runs.
    """
    db = await get_database()
    runs = await db.get_all_runs()
    
    # Paginate
    total = len(runs)
    runs = runs[offset:offset + limit]
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "runs": [r.to_dict() for r in runs],
    }


@app.get("/evaluations/{run_id}", tags=["Evaluations"])
async def get_evaluation(run_id: str) -> dict[str, Any]:
    """
    Get evaluation run details.
    """
    db = await get_database()
    run = await db.get_run(run_id)
    
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    
    return run.to_dict()


@app.get("/evaluations/{run_id}/status", tags=["Evaluations"])
async def get_evaluation_status(run_id: str) -> dict[str, Any]:
    """
    Get evaluation run status and progress.
    """
    db = await get_database()
    run = await db.get_run(run_id)
    
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    
    is_running = run_id in app.state.active_runs
    
    return {
        "run_id": run_id,
        "status": run.status,
        "is_running": is_running,
        "total_prompts": run.total_prompts,
        "successful": run.successful_evaluations,
        "failed": run.failed_evaluations,
        "progress": (
            (run.successful_evaluations + run.failed_evaluations) / run.total_prompts * 100
            if run.total_prompts > 0 else 0
        ),
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    }


@app.post("/evaluations/{run_id}/cancel", tags=["Evaluations"])
async def cancel_evaluation(run_id: str) -> dict[str, Any]:
    """
    Cancel a running evaluation.
    """
    if run_id not in app.state.active_runs:
        raise HTTPException(status_code=404, detail="Run not found or not running")
    
    task = app.state.active_runs[run_id]
    task.cancel()
    
    db = await get_database()
    await db.update_run(run_id, status="cancelled")
    
    return {"message": f"Evaluation {run_id} cancelled"}


# =============================================================================
# Results Endpoints
# =============================================================================


@app.get("/evaluations/{run_id}/results", tags=["Results"])
async def get_results(
    run_id: str,
    model: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """
    Get evaluation results.
    """
    db = await get_database()
    run = await db.get_run(run_id)
    
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    
    records = await db.get_results(run_id, model)
    
    # Paginate
    total = len(records)
    records = records[offset:offset + limit]
    
    return {
        "run_id": run_id,
        "total": total,
        "limit": limit,
        "offset": offset,
        "results": [r.to_dict() for r in records],
    }


@app.get("/evaluations/{run_id}/summary", tags=["Results"])
async def get_results_summary(run_id: str) -> dict[str, Any]:
    """
    Get aggregated summary statistics for a run.
    """
    db = await get_database()
    run = await db.get_run(run_id)
    
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    
    summary = await db.get_results_summary(run_id)
    
    if summary.get("error"):
        raise HTTPException(status_code=500, detail=summary["error"])
    
    return summary


@app.get("/evaluations/{run_id}/export/{format}", tags=["Results"])
async def export_results(
    run_id: str,
    format: str,
) -> StreamingResponse:
    """
    Export evaluation results in various formats.
    Supported formats: csv, json
    """
    if format not in ["csv", "json"]:
        raise HTTPException(status_code=400, detail="Unsupported format. Use csv or json.")
    
    db = await get_database()
    run = await db.get_run(run_id)
    
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    
    # Generate export
    if format == "csv":
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        await db.export_to_csv(run_id, temp_path)
        
        with open(temp_path, 'r') as f:
            content = f.read()
        
        os.unlink(temp_path)
        
        return StreamingResponse(
            iter([content]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={run_id}.csv"},
        )
    
    else:  # json
        df = await db.export_to_dataframe(run_id)
        content = df.to_json(orient="records", indent=2)
        
        return StreamingResponse(
            iter([content]),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={run_id}.json"},
        )


# =============================================================================
# Startup & Shutdown
# =============================================================================


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting AI Evaluation Pipeline API...")
    
    # Initialize database
    await get_database()
    
    logger.info("API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down AI Evaluation Pipeline API...")
    
    # Cancel active runs
    for run_id, task in app.state.active_runs.items():
        logger.info(f"Cancelling run: {run_id}")
        task.cancel()
    
    # Close database
    await close_database()
    
    logger.info("API shutdown complete")


# =============================================================================
# Error Handlers
# =============================================================================


@app.exception_handler(EvaluationPipelineError)
async def pipeline_error_handler(request, exc: EvaluationPipelineError):
    """Handle pipeline-specific errors."""
    return JSONResponse(
        status_code=400,
        content={"detail": exc.message, "type": exc.__class__.__name__},
    )


@app.exception_handler(Exception)
async def general_error_handler(request, exc: Exception):
    """Handle unexpected errors."""
    logger.exception(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "InternalError"},
    )