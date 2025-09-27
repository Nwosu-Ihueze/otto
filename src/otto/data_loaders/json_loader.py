import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from otto.data_loaders.base_loader import BaseLoader
from otto.utils.documents.document_utils import Document


class JSONLoader(BaseLoader):
    """
    Load JSON files without requiring jq dependency.
    
    Handles both regular JSON files and JSON Lines format.
    Can extract specific fields or load entire objects as documents.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        content_key: Optional[str] = None,
        metadata_keys: Optional[List[str]] = None,
        json_lines: bool = False,
        encoding: str = "utf-8",
    ):
        """
        Initialize the Simple JSON Loader.

        Args:
            file_path: Path to the JSON file
            content_key: Key to extract as document content. If None, uses entire object
            metadata_keys: List of keys to include in metadata. If None, includes all
            json_lines: Whether the file is in JSON Lines format (one JSON per line)
            encoding: File encoding to use
        """
        self.file_path = Path(file_path).resolve()
        self.content_key = content_key
        self.metadata_keys = metadata_keys or []
        self.json_lines = json_lines
        self.encoding = encoding

    def lazy_load(self) -> Iterator[Document]:
        """Load and return documents from the JSON file."""
        if self.json_lines:
            yield from self._load_jsonl()
        else:
            yield from self._load_json()

    def _load_json(self) -> Iterator[Document]:
        """Load regular JSON file."""
        try:
            with self.file_path.open(encoding=self.encoding) as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # Array of objects
                for i, item in enumerate(data):
                    yield self._create_document(item, i)
            elif isinstance(data, dict):
                # Single object - check if it contains arrays
                yield from self._process_dict(data)
            else:
                # Primitive value
                yield self._create_document(data, 0)
                
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {self.file_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading JSON file {self.file_path}: {e}")

    def _load_jsonl(self) -> Iterator[Document]:
        """Load JSON Lines file."""
        try:
            with self.file_path.open(encoding=self.encoding) as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            data = json.loads(line)
                            yield self._create_document(data, i)
                        except json.JSONDecodeError as e:
                            # Log warning but continue processing
                            print(f"Warning: Invalid JSON on line {i+1}: {e}")
                            continue
        except Exception as e:
            raise RuntimeError(f"Error loading JSON Lines file {self.file_path}: {e}")

    def _process_dict(self, data: Dict[str, Any]) -> Iterator[Document]:
        """Process a dictionary, looking for arrays to iterate over."""
        # First, try to find arrays in the dictionary
        arrays_found = []
        for key, value in data.items():
            if isinstance(value, list) and value:  # Non-empty list
                arrays_found.append((key, value))
        
        if arrays_found:
            # If we have arrays, iterate over the first one
            # (or we could iterate over all of them)
            key, array = arrays_found[0]
            for i, item in enumerate(array):
                # Include the parent context in metadata
                metadata = {"parent_key": key, "source": str(self.file_path)}
                if isinstance(item, dict):
                    # Add other top-level keys as context
                    for k, v in data.items():
                        if k != key and not isinstance(v, (list, dict)):
                            metadata[f"context_{k}"] = v
                
                yield self._create_document_with_metadata(item, i, metadata)
        else:
            # No arrays found, treat the whole dict as one document
            yield self._create_document(data, 0)

    def _create_document(self, data: Any, index: int) -> Document:
        """Create a document from data with default metadata."""
        return self._create_document_with_metadata(data, index, {})

    def _create_document_with_metadata(
        self, 
        data: Any, 
        index: int, 
        extra_metadata: Dict[str, Any]
    ) -> Document:
        """Create a document with custom metadata."""
        # Extract content
        if self.content_key and isinstance(data, dict):
            content = data.get(self.content_key, "")
        else:
            content = data

        # Convert content to string
        if isinstance(content, str):
            page_content = content
        elif isinstance(content, (dict, list)):
            page_content = json.dumps(content, indent=2)
        else:
            page_content = str(content) if content is not None else ""

        # Build metadata
        metadata = {
            "source": str(self.file_path),
            "seq_num": index,
            **extra_metadata
        }

        # Add specified metadata keys
        if isinstance(data, dict):
            if self.metadata_keys:
                # Only include specified keys
                for key in self.metadata_keys:
                    if key in data:
                        metadata[key] = data[key]
            else:
                # Include all keys except content_key
                for key, value in data.items():
                    if key != self.content_key:
                        # Only include simple values in metadata
                        if isinstance(value, (str, int, float, bool, type(None))):
                            metadata[key] = value

        return Document(page_content=page_content, metadata=metadata)


# Simple factory function to create appropriate loader
def create_json_loader(file_path: Union[str, Path], **kwargs) -> JSONLoader:
    """
    Create a JSON loader with sensible defaults.
    
    Args:
        file_path: Path to JSON file
        **kwargs: Additional arguments for SimpleJSONLoader
        
    Returns:
        Configured SimpleJSONLoader instance
    """
    # Auto-detect JSON Lines format from file extension
    path = Path(file_path)
    if path.suffix.lower() in ('.jsonl', '.ndjson'):
        kwargs.setdefault('json_lines', True)
    
    return JSONLoader(file_path, **kwargs)