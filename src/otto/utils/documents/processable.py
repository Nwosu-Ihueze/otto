import os
import tempfile
import shutil
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass

from otto.utils.archives import is_archive, ArchiveType, infer_archive_type
from otto.utils.file_detector import FileTypeDetector

logger = logging.getLogger(__name__)


@dataclass
class ProcessableFile:
    """Represents a file ready for processing."""
    file_path: str
    mime_type: str
    source_archive: Optional[str] = None  # Path to source archive if extracted
    relative_path: Optional[str] = None   # Relative path within archive


class ProcessedFileSet:
    """
    Manages extracted files and temporary directories for processing pipeline.
    
    Separates the concern of "what was uploaded" from "what needs to be processed".
    Handles archive extraction, nested archives, and temporary file cleanup.
    """
    
    def __init__(self, source_metadata, file_type_detector: Optional[FileTypeDetector] = None):
        """
        Initialize with source file metadata.
        
        Args:
            source_metadata: UploadedFileMetadata for the source file
            file_type_detector: FileTypeDetector instance (creates new one if None)
        """
        self.source_metadata = source_metadata
        self.file_type_detector = file_type_detector or FileTypeDetector()
        
        # Temporary directory for extractions
        self.temp_dir: Optional[str] = None
        self._temp_dir_created = False
        
        # Track all processable files
        self.processable_files: List[ProcessableFile] = []
        
        # Track extracted archives to prevent infinite recursion
        self._extracted_archives: Set[str] = set()
        
        # Track whether extraction has been performed
        self._extraction_performed = False
    
    def extract_if_archive(self, max_depth: int = 3) -> bool:
        """
        Extract source file if it's an archive, handling nested archives.
        
        Args:
            max_depth: Maximum recursion depth for nested archives
            
        Returns:
            True if extraction was performed, False otherwise
        """
        if self._extraction_performed:
            logger.warning("Extraction already performed for this file set")
            return False
        
        self._extraction_performed = True
        
        # Check if source file is an archive
        if not is_archive(self.source_metadata.file_path):
            # Not an archive, add as single processable file
            mime_type = self.file_type_detector.detect_mime_type(
                self.source_metadata.file_path, 
                self.source_metadata.mime_type
            )
            self.processable_files.append(ProcessableFile(
                file_path=self.source_metadata.file_path,
                mime_type=mime_type
            ))
            return False
        
        # Create temporary directory for extraction
        self._create_temp_dir()
        
        try:
            # Extract the archive
            extracted_files = self._extract_archive_to_temp(
                self.source_metadata.file_path,
                max_depth=max_depth
            )
            
            logger.info(f"Extracted {len(extracted_files)} files from {self.source_metadata.original_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract archive {self.source_metadata.original_filename}: {e}")
            self.cleanup()
            raise
    
    def _create_temp_dir(self) -> None:
        """Create temporary directory for extractions."""
        if not self._temp_dir_created:
            self.temp_dir = tempfile.mkdtemp(prefix="otto_extract_")
            self._temp_dir_created = True
            logger.debug(f"Created temporary directory: {self.temp_dir}")
    
    def _extract_archive_to_temp(self, archive_path: str, max_depth: int, current_depth: int = 0) -> List[str]:
        """
        Recursively extract archives to temporary directory.
        
        Args:
            archive_path: Path to archive file
            max_depth: Maximum recursion depth
            current_depth: Current recursion depth
            
        Returns:
            List of extracted file paths
        """
        if current_depth >= max_depth:
            logger.warning(f"Maximum extraction depth ({max_depth}) reached for {archive_path}")
            return []
        
        if archive_path in self._extracted_archives:
            logger.warning(f"Archive {archive_path} already extracted, skipping to prevent loops")
            return []
        
        self._extracted_archives.add(archive_path)
        
        # Determine extraction directory
        archive_name = Path(archive_path).stem
        extract_dir = os.path.join(self.temp_dir, f"extract_{current_depth}_{archive_name}")
        os.makedirs(extract_dir, exist_ok=True)
        
        try:
            # Extract archive using the existing extract_archive function
            # But we need to modify it to extract to our specific directory
            extracted_files = self._safe_extract_archive(archive_path, extract_dir)
            
            all_extracted = []
            
            # Process each extracted file
            for extracted_file in extracted_files:
                if extracted_file.startswith('__MACOSX/') or extracted_file.startswith('._'):
                    logger.debug(f"Skipping macOS metadata file: {extracted_file}")
                    continue
                
                full_path = os.path.join(extract_dir, extracted_file)
                
                if not os.path.exists(full_path):
                    logger.warning(f"Extracted file not found: {full_path}")
                    continue
                
                # Check if extracted file is also an archive
                if is_archive(full_path) and current_depth < max_depth:
                    # Recursively extract nested archive
                    nested_extracted = self._extract_archive_to_temp(
                        full_path, max_depth, current_depth + 1
                    )
                    all_extracted.extend(nested_extracted)
                else:
                    # Add as processable file
                    mime_type = self.file_type_detector.detect_mime_type(full_path)
                    
                    # Skip if mime type is not supported
                    try:
                        self.file_type_detector.get_loader_class(full_path, mime_type)
                        
                        self.processable_files.append(ProcessableFile(
                            file_path=full_path,
                            mime_type=mime_type,
                            source_archive=archive_path,
                            relative_path=extracted_file
                        ))
                        all_extracted.append(full_path)
                        
                    except ValueError as e:
                        logger.debug(f"Skipping unsupported file {full_path}: {e}")
            
            return all_extracted
            
        except Exception as e:
            logger.error(f"Failed to extract {archive_path}: {e}")
            return []
    
    def _safe_extract_archive(self, archive_path: str, extract_dir: str) -> List[str]:
        """
        Safely extract archive to specified directory.
        
        Args:
            archive_path: Path to archive file
            extract_dir: Directory to extract to
            
        Returns:
            List of relative paths of extracted files
        """
        archive_type = infer_archive_type(archive_path)
        
        if archive_type == ArchiveType.ZIP:
            return self._extract_zip(archive_path, extract_dir)
        elif archive_type == ArchiveType.GZIP:
            return self._extract_gzip(archive_path, extract_dir)
        elif archive_type in {ArchiveType.TAR, ArchiveType.TAR_GZ, ArchiveType.TAR_BZ2}:
            return self._extract_tar(archive_path, extract_dir)
        else:
            raise ValueError(f"Unsupported archive type: {archive_type}")
    
    def _extract_zip(self, archive_path: str, extract_dir: str) -> List[str]:
        """Extract ZIP archive."""
        import zipfile
        
        extracted_files = []
        with zipfile.ZipFile(archive_path, 'r') as zip_file:
            for member in zip_file.namelist():
                # Security check for path traversal
                if os.path.isabs(member) or ".." in member:
                    logger.warning(f"Skipping potentially dangerous path: {member}")
                    continue
                
                try:
                    zip_file.extract(member, extract_dir)
                    extracted_files.append(member)
                except Exception as e:
                    logger.warning(f"Failed to extract {member}: {e}")
        
        return extracted_files
    
    def _extract_gzip(self, archive_path: str, extract_dir: str) -> List[str]:
        """Extract GZIP archive."""
        import gzip
        
        # For .gz files, the extracted filename is usually the original without .gz
        extracted_name = Path(archive_path).stem
        extracted_path = os.path.join(extract_dir, extracted_name)
        
        with gzip.open(archive_path, 'rb') as gz_file:
            with open(extracted_path, 'wb') as output_file:
                shutil.copyfileobj(gz_file, output_file)
        
        return [extracted_name]
    
    def _extract_tar(self, archive_path: str, extract_dir: str) -> List[str]:
        """Extract TAR archive (including .tar.gz, .tar.bz2)."""
        import tarfile
        
        extracted_files = []
        with tarfile.open(archive_path, 'r:*') as tar_file:
            for member in tar_file.getmembers():
                
                if os.path.isabs(member.name) or ".." in member.name:
                    logger.warning(f"Skipping potentially dangerous path: {member.name}")
                    continue
                
                if member.isfile():
                    try:
                        tar_file.extract(member, extract_dir)
                        extracted_files.append(member.name)
                    except Exception as e:
                        logger.warning(f"Failed to extract {member.name}: {e}")
        
        return extracted_files
    
    def get_processable_files(self) -> List[ProcessableFile]:
        """
        Get list of files ready for processing.
        
        Returns:
            List of ProcessableFile objects
        """
        if not self._extraction_performed:
            logger.warning("extract_if_archive() has not been called yet")
        
        return self.processable_files.copy()
    
    def get_file_paths_and_types(self) -> List[Tuple[str, str]]:
        """
        Get list of (file_path, mime_type) tuples for processing.
        
        Returns:
            List of (file_path, mime_type) tuples
        """
        return [(pf.file_path, pf.mime_type) for pf in self.processable_files]
    
    def has_files(self) -> bool:
        """Check if there are any processable files."""
        return len(self.processable_files) > 0
    
    def cleanup(self) -> None:
        """Clean up temporary directories and files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.debug(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.error(f"Failed to clean up temporary directory {self.temp_dir}: {e}")
            finally:
                self.temp_dir = None
                self._temp_dir_created = False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
    
    def __del__(self):
        """Destructor with cleanup."""
        self.cleanup()