"""
Production Entry Point for AI Evaluation Pipeline API.
Includes graceful shutdown, structured logging, and metrics.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from uvicorn.config import Config

from src.storage.database import get_database, close_database

# =============================================================================
# Structured Logging Configuration
# =============================================================================

class StructuredFormatter(logging.Formatter):
    """JSON-structured logging formatter for production."""
    
    def __init__(self) -> None:
        super().__init__()
        self.hostname = _get_hostname()
    
    def format(self, record: logging.LogRecord) -> str:
        import json
        from datetime import datetime, timezone
        
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "hostname": self.hostname,
            "process_id": record.process,
            "thread_id": record.thread,
        }
        
        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


def _get_hostname() -> str:
    """Get the hostname for logging."""
    import socket
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured logging for production."""
    import logging.handlers
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with structured format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(console_handler)
    
    # File handler with detailed format
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "evaluation.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    root_logger.addHandler(file_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


# =============================================================================
# Graceful Shutdown Handler
# =============================================================================

class GracefulShutdown:
    """Handle graceful shutdown of the application."""
    
    def __init__(self) -> None:
        self.shutdown_event = asyncio.Event()
        self.force_shutdown = False
    
    def signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals."""
        import logging
        logger = logging.getLogger(__name__)
        
        signal_names = {signal.SIGTERM: "SIGTERM", signal.SIGINT: "SIGINT"}
        sig_name = signal_names.get(signum, str(signum))
        
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")
        
        if self.force_shutdown:
            logger.warning("Force shutdown requested, exiting immediately")
            sys.exit(1)
        
        self.force_shutdown = True
        self.shutdown_event.set()
    
    def setup(self) -> None:
        """Set up signal handlers."""
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
    
    async def wait(self) -> None:
        """Wait for shutdown signal."""
        await self.shutdown_event.wait()


# =============================================================================
# Application Lifecycle
# =============================================================================

shutdown_handler = GracefulShutdown()


@asynccontextmanager
async def lifespan_manager(app):
    """
    Manage application lifecycle.
    Handles startup and shutdown operations.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("AI Evaluation Pipeline - Starting up")
    logger.info("=" * 60)
    
    # Startup
    try:
        # Initialize database
        logger.info("Initializing database connection...")
        db = await get_database()
        logger.info("Database connection established")
        
        # Warm up models cache
        logger.info("Warming up model configurations...")
        from src.core.config import get_config
        config = get_config()
        models = config.get_all_models()
        logger.info(f"Loaded {len(models)} model configurations")
        
        logger.info("Application startup complete")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("=" * 60)
    logger.info("AI Evaluation Pipeline - Shutting down")
    logger.info("=" * 60)
    
    try:
        # Close database connections
        logger.info("Closing database connections...")
        await close_database()
        logger.info("Database connections closed")
        
        # Give time for cleanup
        await asyncio.sleep(0.5)
        
        logger.info("Application shutdown complete")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


# =============================================================================
# Metrics Middleware (Prometheus-style)
# =============================================================================

class MetricsMiddleware:
    """Middleware for collecting request metrics."""
    
    def __init__(self, app) -> None:
        self.app = app
        self.request_count = 0
        self.request_duration: list[float] = []
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        import time
        from starlette.types import ASGIApp
        
        start_time = time.perf_counter()
        self.request_count += 1
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # Record metrics
                duration = time.perf_counter() - start_time
                self.request_duration.append(duration)
                
                # Keep only last 1000 measurements
                if len(self.request_duration) > 1000:
                    self.request_duration = self.request_duration[-1000:]
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)


# =============================================================================
# Main Entry Point
# =============================================================================

def run_api(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    log_level: str = "INFO",
    reload: bool = False,
) -> None:
    """
    Run the API server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        workers: Number of worker processes
        log_level: Logging level
        reload: Enable auto-reload (development only)
    """
    # Setup logging
    setup_logging(log_level)
    
    import logging
    logger = logging.getLogger(__name__)
    
    # Configure uvicorn
    config = Config(
        app="src.api.routes:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        reload=reload,
        lifespan="on",
        access_log=True,
        use_colors=True,
    )
    
    # Setup shutdown handler
    shutdown_handler.setup()
    
    logger.info(f"Starting API server on {host}:{port}")
    logger.info(f"Workers: {workers}")
    logger.info(f"Log level: {log_level}")
    
    try:
        server = uvicorn.Server(config)
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        logger.info("Server stopped")


def run_dashboard(
    host: str = "0.0.0.0",
    port: int = 8501,
    log_level: str = "INFO",
) -> None:
    """
    Run the Streamlit dashboard.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        log_level: Logging level
    """
    import subprocess
    
    setup_logging(log_level)
    
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting dashboard on {host}:{port}")
    
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "src/dashboard/app.py",
        "--server.address", host,
        "--server.port", str(port),
        "--browser.gatherUsageStats", "false",
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Dashboard error: {e}")
        raise


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Evaluation Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # API command
    api_parser = subparsers.add_parser("api", help="Run API server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    api_parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    api_parser.add_argument("--log-level", default="INFO", help="Logging level")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Run dashboard")
    dashboard_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    dashboard_parser.add_argument("--port", type=int, default=8501, help="Port to bind to")
    dashboard_parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    if args.command == "api":
        run_api(
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level=args.log_level,
            reload=args.reload,
        )
    elif args.command == "dashboard":
        run_dashboard(
            host=args.host,
            port=args.port,
            log_level=args.log_level,
        )
    else:
        parser.print_help()