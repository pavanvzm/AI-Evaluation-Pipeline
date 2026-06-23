"""
Data Ingestion Module for AI Evaluation Pipeline.
Handles loading, validation, and management of benchmark datasets.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from src.core.exceptions import DatasetError, DatasetValidationError

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class DatasetItem(BaseModel):
    """Represents a single item in a benchmark dataset."""
    id: str = Field(..., description="Unique identifier for the item")
    prompt: str = Field(..., min_length=1, description="The input prompt text")
    ground_truth: str | None = Field(default=None, description="Expected correct response")
    context: str | None = Field(default=None, description="Supporting context for the prompt")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator("prompt")
    @classmethod
    def validate_prompt_not_empty(cls, v: str) -> str:
        """Ensure prompt is not just whitespace."""
        if not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace only")
        return v.strip()


class DatasetMetadata(BaseModel):
    """Metadata about a dataset."""
    name: str
    file_path: str
    total_items: int
    columns: list[str]
    has_ground_truth: bool
    has_context: bool
    created_at: str | None = None
    version: str = "1.0"


@dataclass
class Dataset:
    """
    Represents a benchmark dataset for LLM evaluation.
    """
    name: str
    items: list[DatasetItem] = field(default_factory=list)
    metadata: DatasetMetadata | None = None
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __iter__(self) -> Iterator[DatasetItem]:
        return iter(self.items)
    
    def __getitem__(self, index: int) -> DatasetItem:
        return self.items[index]
    
    def get_items_with_ground_truth(self) -> list[DatasetItem]:
        """Get items that have ground truth responses."""
        return [item for item in self.items if item.ground_truth is not None]
    
    def get_items_with_context(self) -> list[DatasetItem]:
        """Get items that have context."""
        return [item for item in self.items if item.context is not None]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert dataset to pandas DataFrame."""
        data = []
        for item in self.items:
            row = {
                "id": item.id,
                "prompt": item.prompt,
                "ground_truth": item.ground_truth,
                "context": item.context,
            }
            row.update(item.metadata)
            data.append(row)
        return pd.DataFrame(data)
    
    def summary(self) -> dict[str, Any]:
        """Get a summary of the dataset."""
        return {
            "name": self.name,
            "total_items": len(self.items),
            "items_with_ground_truth": len(self.get_items_with_ground_truth()),
            "items_with_context": len(self.get_items_with_context()),
            "avg_prompt_length": sum(len(item.prompt) for item in self.items) / len(self.items) if self.items else 0,
            "avg_context_length": sum(len(item.context or "") for item in self.items) / len(self.items) if self.items else 0,
        }


# =============================================================================
# Dataset Validators
# =============================================================================


class DatasetValidator:
    """
    Validates dataset files and their contents.
    """
    
    REQUIRED_COLUMNS = {"id", "prompt"}
    OPTIONAL_COLUMNS = {"ground_truth", "context", "metadata"}
    ALL_COLUMNS = REQUIRED_COLUMNS | OPTIONAL_COLUMNS
    
    @classmethod
    def validate_file(cls, file_path: str | Path) -> tuple[bool, str]:
        """
        Validate that a file exists and has a supported format.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        path = Path(file_path)
        
        if not path.exists():
            return False, f"File not found: {file_path}"
        
        if not path.is_file():
            return False, f"Path is not a file: {file_path}"
        
        supported_formats = {".csv", ".json", ".jsonl"}
        if path.suffix.lower() not in supported_formats:
            return False, f"Unsupported file format: {path.suffix}. Supported formats: {supported_formats}"
        
        return True, ""
    
    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> tuple[bool, str, list[str]]:
        """
        Validate a pandas DataFrame has the required columns.
        
        Returns:
            Tuple of (is_valid, error_message, missing_columns)
        """
        columns = set(df.columns)
        missing_required = cls.REQUIRED_COLUMNS - columns
        
        if missing_required:
            return False, f"Missing required columns: {missing_required}", list(missing_required)
        
        # Check for empty dataframe
        if df.empty:
            return False, "Dataset is empty", []
        
        # Check for null values in required columns
        null_prompts = df["prompt"].isnull().sum()
        null_ids = df["id"].isnull().sum()
        
        if null_prompts > 0:
            return False, f"Found {null_prompts} null values in 'prompt' column", []
        
        if null_ids > 0:
            return False, f"Found {null_ids} null values in 'id' column", []
        
        return True, "", []
    
    @classmethod
    def validate_dataset(cls, dataset: Dataset) -> tuple[bool, str]:
        """
        Validate a Dataset object.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(dataset) == 0:
            return False, "Dataset is empty"
        
        # Check for duplicate IDs
        ids = [item.id for item in dataset]
        if len(ids) != len(set(ids)):
            duplicates = [id_ for id_ in ids if ids.count(id_) > 1]
            return False, f"Found duplicate IDs: {set(duplicates)}"
        
        return True, ""


# =============================================================================
# Dataset Loader
# =============================================================================


class DatasetLoader:
    """
    Loads datasets from various file formats.
    """
    
    @staticmethod
    def load(file_path: str | Path, name: str | None = None) -> Dataset:
        """
        Load a dataset from a file.
        
        Args:
            file_path: Path to the dataset file
            name: Optional name for the dataset
            
        Returns:
            Dataset object
            
        Raises:
            DatasetError: If file cannot be loaded
            DatasetValidationError: If file format is invalid
        """
        path = Path(file_path)
        
        # Validate file
        is_valid, error = DatasetValidator.validate_file(path)
        if not is_valid:
            raise DatasetValidationError(error)
        
        # Determine format and load
        suffix = path.suffix.lower()
        
        try:
            if suffix == ".csv":
                return DatasetLoader._load_csv(path, name or path.stem)
            elif suffix == ".json":
                return DatasetLoader._load_json(path, name or path.stem)
            elif suffix == ".jsonl":
                return DatasetLoader._load_jsonl(path, name or path.stem)
            else:
                raise DatasetValidationError(f"Unsupported format: {suffix}")
        except DatasetValidationError:
            raise
        except Exception as e:
            raise DatasetError(f"Failed to load dataset: {e}")
    
    @staticmethod
    def _load_csv(path: Path, name: str) -> Dataset:
        """Load dataset from CSV file."""
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise DatasetValidationError(f"Failed to parse CSV: {e}")
        
        # Validate DataFrame
        is_valid, error, _ = DatasetValidator.validate_dataframe(df)
        if not is_valid:
            raise DatasetValidationError(error)
        
        # Convert to DatasetItem objects
        items = []
        for idx, row in df.iterrows():
            item_data = {
                "id": str(row["id"]),
                "prompt": str(row["prompt"]),
            }
            
            if "ground_truth" in row and pd.notna(row["ground_truth"]):
                item_data["ground_truth"] = str(row["ground_truth"])
            
            if "context" in row and pd.notna(row["context"]):
                item_data["context"] = str(row["context"])
            
            if "metadata" in row and pd.notna(row["metadata"]):
                try:
                    item_data["metadata"] = json.loads(str(row["metadata"]))
                except json.JSONDecodeError:
                    item_data["metadata"] = {"raw": str(row["metadata"])}
            
            items.append(DatasetItem(**item_data))
        
        # Create metadata
        metadata = DatasetMetadata(
            name=name,
            file_path=str(path),
            total_items=len(items),
            columns=list(df.columns),
            has_ground_truth="ground_truth" in df.columns,
            has_context="context" in df.columns,
        )
        
        return Dataset(name=name, items=items, metadata=metadata)
    
    @staticmethod
    def _load_json(path: Path, name: str) -> Dataset:
        """Load dataset from JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise DatasetValidationError(f"Invalid JSON: {e}")
        except Exception as e:
            raise DatasetError(f"Failed to read JSON file: {e}")
        
        # Handle both array and object formats
        if isinstance(data, list):
            items_data = data
        elif isinstance(data, dict):
            items_data = data.get("items", [data])
        else:
            raise DatasetValidationError("JSON must be an array of objects or have an 'items' key")
        
        # Convert to DatasetItem objects
        items = []
        for idx, item_data in enumerate(items_data):
            try:
                # Ensure required fields exist
                if "id" not in item_data:
                    item_data["id"] = f"item_{idx}"
                if "prompt" not in item_data:
                    raise DatasetValidationError(f"Item {idx} missing required 'prompt' field")
                
                # Parse metadata if present as string
                if "metadata" in item_data and isinstance(item_data["metadata"], str):
                    try:
                        item_data["metadata"] = json.loads(item_data["metadata"])
                    except json.JSONDecodeError:
                        item_data["metadata"] = {"raw": item_data["metadata"]}
                
                items.append(DatasetItem(**item_data))
            except Exception as e:
                raise DatasetValidationError(f"Invalid item at index {idx}: {e}")
        
        if not items:
            raise DatasetValidationError("No valid items found in JSON")
        
        # Create metadata
        all_keys = set()
        for item in items:
            all_keys.update(item.model_dump().keys())
        
        metadata = DatasetMetadata(
            name=name,
            file_path=str(path),
            total_items=len(items),
            columns=list(all_keys),
            has_ground_truth=any(item.ground_truth is not None for item in items),
            has_context=any(item.context is not None for item in items),
        )
        
        return Dataset(name=name, items=items, metadata=metadata)
    
    @staticmethod
    def _load_jsonl(path: Path, name: str) -> Dataset:
        """Load dataset from JSONL (JSON Lines) file."""
        items = []
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        item_data = json.loads(line)
                        
                        # Ensure required fields exist
                        if "id" not in item_data:
                            item_data["id"] = f"item_{idx}"
                        if "prompt" not in item_data:
                            raise DatasetValidationError(f"Line {idx} missing required 'prompt' field")
                        
                        # Parse metadata if present as string
                        if "metadata" in item_data and isinstance(item_data["metadata"], str):
                            try:
                                item_data["metadata"] = json.loads(item_data["metadata"])
                            except json.JSONDecodeError:
                                item_data["metadata"] = {"raw": item_data["metadata"]}
                        
                        items.append(DatasetItem(**item_data))
                    except json.JSONDecodeError as e:
                        raise DatasetValidationError(f"Invalid JSON at line {idx}: {e}")
        except DatasetValidationError:
            raise
        except Exception as e:
            raise DatasetError(f"Failed to read JSONL file: {e}")
        
        if not items:
            raise DatasetValidationError("No valid items found in JSONL file")
        
        # Create metadata
        all_keys = set()
        for item in items:
            all_keys.update(item.model_dump().keys())
        
        metadata = DatasetMetadata(
            name=name,
            file_path=str(path),
            total_items=len(items),
            columns=list(all_keys),
            has_ground_truth=any(item.ground_truth is not None for item in items),
            has_context=any(item.context is not None for item in items),
        )
        
        return Dataset(name=name, items=items, metadata=metadata)


# =============================================================================
# Dataset Utilities
# =============================================================================


def load_dataset(file_path: str | Path, name: str | None = None) -> Dataset:
    """
    Convenience function to load a dataset.
    
    Args:
        file_path: Path to the dataset file
        name: Optional name for the dataset
        
    Returns:
        Dataset object
    """
    return DatasetLoader.load(file_path, name)


def create_sample_dataset() -> Dataset:
    """
    Create a sample dataset for testing.
    
    Returns:
        Dataset with sample customer support items
    """
    items = [
        DatasetItem(
            id="q1",
            prompt="What is your return policy?",
            context="Our return policy allows returns within 30 days of purchase with original receipt. Items must be in original condition with tags attached.",
            ground_truth="You can return items within 30 days of purchase with your original receipt. Items should be in their original condition with tags still attached."
        ),
        DatasetItem(
            id="q2",
            prompt="How do I reset my password?",
            context="To reset your password, click on 'Forgot Password' on the login page. Enter your email address and we'll send you a link to create a new password.",
            ground_truth="Click 'Forgot Password' on the login page, enter your email, and follow the link we send to create a new password."
        ),
        DatasetItem(
            id="q3",
            prompt="What are your business hours?",
            context="We are open Monday through Friday from 9 AM to 6 PM EST. We are closed on weekends and major holidays.",
            ground_truth="Our hours are Monday to Friday, 9 AM to 6 PM EST. We're closed weekends and holidays."
        ),
        DatasetItem(
            id="q4",
            prompt="How can I contact support?",
            context="You can reach our support team via email at support@example.com or by phone at 1-800-555-0123. Live chat is available during business hours.",
            ground_truth="Contact us at support@example.com, call 1-800-555-0123, or use our live chat during business hours."
        ),
        DatasetItem(
            id="q5",
            prompt="Do you offer international shipping?",
            context="We ship to over 50 countries worldwide. International shipping takes 7-14 business days. Shipping costs vary by destination.",
            ground_truth="Yes, we ship to 50+ countries. International delivery takes 7-14 business days with varying costs."
        ),
    ]
    
    return Dataset(
        name="sample_customer_support",
        items=items,
        metadata=DatasetMetadata(
            name="sample_customer_support",
            file_path="in_memory",
            total_items=5,
            columns=["id", "prompt", "ground_truth", "context"],
            has_ground_truth=True,
            has_context=True,
        )
    )