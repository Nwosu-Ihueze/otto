import mimetypes
import os
from pathlib import Path
from typing import Dict, Type, Optional, Union
# import magic  # python-magic library
import logging

from otto.data_loaders.base_loader import BaseLoader
from otto.data_loaders.csv_loader import CSVLoader
from otto.data_loaders.json_loader import JSONLoader
from otto.data_loaders.text_loader import TextLoader



logger = logging.getLogger(__name__)

try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False
    logger.warning("python-magic not available, using fallback detection methods")

class FileTypeDetector:
    """Service for detecting file types and mapping to appropriate loaders."""
    
    def __init__(self):
        # MIME type to loader class mapping
        self._mime_to_loader: Dict[str, Type[BaseLoader]] = {
            'text/csv': CSVLoader,
            'text/tab-separated-values': CSVLoader,  # TSV files
            'application/json': JSONLoader,
            'text/json': JSONLoader,
            'text/plain': TextLoader,
        }
        
        # File extension to MIME type mapping (fallback)
        self._extension_to_mime: Dict[str, str] = {
            '.csv': 'text/csv',
            '.tsv': 'text/tab-separated-values',
            '.json': 'application/json',
            '.jsonl': 'application/json',
            '.txt': 'text/plain',
            '.md': 'text/plain',
            '.rst': 'text/plain',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.xls': 'application/vnd.ms-excel',
            '.html': 'text/html',
            '.htm': 'text/html',
            '.xml': 'application/xml',
            '.eml': 'message/rfc822',
            '.msg': 'application/vnd.ms-outlook',
        }
        
        # Compound extensions (order matters - check longer extensions first)
        self._compound_extensions = ['.tar.gz', '.tar.bz2', '.tar.xz']
        
        # Initialize magic mime detector if available
        self._magic_mime = None
        if HAS_MAGIC:
            try:
                # Use the already imported magic module
                self._magic_mime = magic.Magic(mime=True)
            except Exception as e:
                logger.warning(f"Could not initialize magic mime detector: {e}")
        else:
            logger.debug("python-magic not available, using fallback detection methods")
    
    def detect_mime_type(self, file_path: Union[str, Path], provided_mime: Optional[str] = None) -> str:
        """
        Detect MIME type using multiple methods with fallback chain.
        
        Args:
            file_path: Path to the file
            provided_mime: MIME type provided by user/upload (if any)
            
        Returns:
            Detected MIME type
        """
        file_path = Path(file_path)
        
       
        if provided_mime and self._is_supported_mime_type(provided_mime):
            logger.debug(f"Using provided MIME type: {provided_mime}")
            return provided_mime
        
       
        if self._magic_mime and file_path.exists():
            try:
                detected_mime = self._magic_mime.from_file(str(file_path))
                if self._is_supported_mime_type(detected_mime):
                    logger.debug(f"Magic detected MIME type: {detected_mime}")
                    return detected_mime
            except Exception as e:
                logger.debug(f"Magic detection failed: {e}")
        
        
        guessed_mime, _ = mimetypes.guess_type(str(file_path))
        if guessed_mime and self._is_supported_mime_type(guessed_mime):
            logger.debug(f"Mimetypes guessed: {guessed_mime}")
            return guessed_mime
        
        extension = self._get_file_extension(file_path)
        if extension in self._extension_to_mime:
            mime_type = self._extension_to_mime[extension]
            logger.debug(f"Extension-based MIME type: {mime_type}")
            return mime_type
        
        if file_path.exists():
            content_mime = self._detect_from_content(file_path)
            if content_mime:
                logger.debug(f"Content-based MIME type: {content_mime}")
                return content_mime
        
        logger.warning(f"Could not detect MIME type for {file_path}, using fallback")
        return 'application/octet-stream'
    
    def get_loader_class(self, file_path: Union[str, Path], mime_type: Optional[str] = None) -> Type[BaseLoader]:
        """
        Get the appropriate loader class for a file.
        
        Args:
            file_path: Path to the file
            mime_type: Known MIME type (optional)
            
        Returns:
            BaseLoader class suitable for the file
            
        Raises:
            ValueError: If no suitable loader is found
        """
        if mime_type is None:
            mime_type = self.detect_mime_type(file_path)
        
        loader_class = self._mime_to_loader.get(mime_type)
        if loader_class:
            return loader_class
        
        
        if mime_type.startswith('text/'):
            logger.info(f"Using TextLoader fallback for MIME type: {mime_type}")
            return TextLoader
        
        raise ValueError(f"No loader available for MIME type: {mime_type} (file: {file_path})")
    
    def create_loader(self, file_path: Union[str, Path], mime_type: Optional[str] = None, **loader_kwargs) -> BaseLoader:
        """
        Create a loader instance for the given file.
        
        Args:
            file_path: Path to the file
            mime_type: Known MIME type (optional)
            **loader_kwargs: Additional arguments to pass to the loader constructor
            
        Returns:
            Configured BaseLoader instance
        """
        loader_class = self.get_loader_class(file_path, mime_type)
        return loader_class(file_path, **loader_kwargs)
    
    def _get_file_extension(self, file_path: Path) -> str:
        """Get file extension, handling compound extensions."""
        file_str = str(file_path).lower()
        
     
        for compound_ext in self._compound_extensions:
            if file_str.endswith(compound_ext):
                return compound_ext
        
       
        return file_path.suffix.lower()
    
    def _is_supported_mime_type(self, mime_type: str) -> bool:
        """Check if a MIME type is supported by our loaders."""
        return mime_type in self._mime_to_loader or mime_type.startswith('text/')
    
    def _detect_from_content(self, file_path: Path) -> Optional[str]:
        """Detect MIME type from file content using heuristics."""
        try:
            
            with open(file_path, 'rb') as f:
                header = f.read(512)
            
         
            if header.strip().startswith((b'{', b'[')):
                try:
                  
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read(1024)
                        if content.strip().startswith(('{', '[')):
                            import json
                            json.loads(content[:100] + '}' if content.startswith('{') else content[:100] + ']')
                            return 'application/json'
                except:
                    pass
            
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if ',' in first_line and len(first_line.split(',')) > 1:
                   
                        return 'text/csv'
            except:
                pass
            
            
            try:
                text_chars = sum(1 for byte in header if 32 <= byte <= 126 or byte in (9, 10, 13))
                if text_chars / len(header) > 0.7: 
                    return 'text/plain'
            except:
                pass
                
        except Exception as e:
            logger.debug(f"Content detection failed for {file_path}: {e}")
        
        return None
    
    def get_supported_mime_types(self) -> list[str]:
        """Get list of all supported MIME types."""
        return list(self._mime_to_loader.keys())
    
    def get_supported_extensions(self) -> list[str]:
        """Get list of all supported file extensions."""
        return list(self._extension_to_mime.keys())