import logging
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass
from enum import Enum

from otto.utils.file_detector import FileTypeDetector
from otto.utils.documents.processable import ProcessedFileSet, ProcessableFile
from otto.utils.documents.document_utils import Document

logger = logging.getLogger(__name__)


class ProcessingStatus(str, Enum):
    """Status of file processing."""
    PENDING = "pending"
    EXTRACTING = "extracting"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessingResult:
    """Result of file processing operation."""
    status: ProcessingStatus
    documents: List[Document]
    processed_files_count: int
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


class FileProcessor:
    """
    Orchestrates the file processing pipeline.
    
    Takes uploaded file metadata and coordinates:
    1. Archive extraction (if needed)
    2. File type detection
    3. Document loading from appropriate loaders
    4. Metadata standardization
    
    This is the main coordinator between upload and preprocessing stages.
    """
    
    def __init__(self, file_type_detector: Optional[FileTypeDetector] = None):
        """
        Initialize the file processor.
        
        Args:
            file_type_detector: FileTypeDetector instance (creates new one if None)
        """
        self.file_type_detector = file_type_detector or FileTypeDetector()
    
    def process_file(
        self, 
        uploaded_metadata, 
        max_extraction_depth: int = 3,
        loader_kwargs: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Process an uploaded file through the complete pipeline.
        
        Args:
            uploaded_metadata: UploadedFileMetadata instance
            max_extraction_depth: Maximum depth for nested archive extraction
            loader_kwargs: Additional kwargs to pass to loaders
            
        Returns:
            ProcessingResult with documents and processing information
        """
        loader_kwargs = loader_kwargs or {}
        
        logger.info(f"Processing file: {uploaded_metadata.original_filename}")
        
        # Initialize result
        result = ProcessingResult(
            status=ProcessingStatus.PENDING,
            documents=[],
            processed_files_count=0,
            errors=[],
            warnings=[],
            metadata={
                "source_file": uploaded_metadata.original_filename,
                "upload_timestamp": uploaded_metadata.upload_timestamp,
                "source_mime_type": uploaded_metadata.mime_type,
                "file_size": uploaded_metadata.file_size
            }
        )
        
        try:
            # Use ProcessedFileSet to handle extraction
            with ProcessedFileSet(uploaded_metadata, self.file_type_detector) as file_set:
                
                # Extract archives if needed
                result.status = ProcessingStatus.EXTRACTING
                was_extracted = file_set.extract_if_archive(max_depth=max_extraction_depth)
                
                if was_extracted:
                    logger.info(f"Archive extraction completed for {uploaded_metadata.original_filename}")
                    result.metadata["was_archive"] = True
                else:
                    result.metadata["was_archive"] = False
                
                # Get processable files
                processable_files = file_set.get_processable_files()
                
                if not processable_files:
                    result.status = ProcessingStatus.FAILED
                    result.errors.append("No processable files found after extraction")
                    return result
                
                result.metadata["total_files"] = len(processable_files)
                
                # Process each file
                result.status = ProcessingStatus.PROCESSING
                documents = []
                
                for processable_file in processable_files:
                    file_result = self._process_single_file(processable_file, loader_kwargs)
                    
                    if file_result.documents:
                        documents.extend(file_result.documents)
                        result.processed_files_count += 1
                    
                    # Collect errors and warnings
                    result.errors.extend(file_result.errors)
                    result.warnings.extend(file_result.warnings)
                
                # Standardize metadata across all documents
                standardized_docs = self._standardize_document_metadata(
                    documents, 
                    uploaded_metadata,
                    processable_files
                )
                
                result.documents = standardized_docs
                result.status = ProcessingStatus.COMPLETED
                
                logger.info(
                    f"Processing completed: {len(standardized_docs)} documents "
                    f"from {result.processed_files_count} files"
                )
                
        except Exception as e:
            logger.error(f"Processing failed for {uploaded_metadata.original_filename}: {e}")
            result.status = ProcessingStatus.FAILED
            result.errors.append(f"Processing error: {str(e)}")
        
        return result
    
    def _process_single_file(
        self, 
        processable_file: ProcessableFile, 
        loader_kwargs: Dict[str, Any]
    ) -> ProcessingResult:
        """
        Process a single file with appropriate loader.
        
        Args:
            processable_file: ProcessableFile instance
            loader_kwargs: Additional kwargs for loaders
            
        Returns:
            ProcessingResult for this single file
        """
        result = ProcessingResult(
            status=ProcessingStatus.PROCESSING,
            documents=[],
            processed_files_count=0,
            errors=[],
            warnings=[],
            metadata={}
        )
        
        try:
            # Get appropriate loader
            loader = self.file_type_detector.create_loader(
                processable_file.file_path,
                processable_file.mime_type,
                **loader_kwargs
            )
            
            # Load documents
            documents = list(loader.lazy_load())
            
            if documents:
                result.documents = documents
                result.processed_files_count = 1
                result.status = ProcessingStatus.COMPLETED
                
                logger.debug(
                    f"Loaded {len(documents)} documents from {processable_file.file_path}"
                )
            else:
                result.warnings.append(f"No documents loaded from {processable_file.file_path}")
                
        except Exception as e:
            logger.error(f"Failed to process {processable_file.file_path}: {e}")
            result.errors.append(f"Error loading {processable_file.file_path}: {str(e)}")
            result.status = ProcessingStatus.FAILED
        
        return result
    
    def _standardize_document_metadata(
        self,
        documents: List[Document],
        uploaded_metadata,
        processable_files: List[ProcessableFile]
    ) -> List[Document]:
        """
        Standardize metadata across all documents.
        
        Args:
            documents: List of loaded documents
            uploaded_metadata: Original upload metadata
            processable_files: List of processed files
            
        Returns:
            Documents with standardized metadata
        """
        # Create file path to processable file mapping
        file_mapping = {pf.file_path: pf for pf in processable_files}
        
        standardized_docs = []
        
        for doc in documents:
            # Copy document
            new_metadata = doc.metadata.copy()
            
            # Add standard fields
            new_metadata.update({
                "upload_id": uploaded_metadata.filename,
                "original_filename": uploaded_metadata.original_filename,
                "upload_timestamp": uploaded_metadata.upload_timestamp.isoformat(),
                "file_checksum": uploaded_metadata.checksum,
                "processing_timestamp": self._get_current_timestamp(),
            })
            
            # Add file-specific metadata if available
            source_path = new_metadata.get("source")
            if source_path and source_path in file_mapping:
                processable_file = file_mapping[source_path]
                new_metadata.update({
                    "detected_mime_type": processable_file.mime_type,
                    "source_archive": processable_file.source_archive,
                    "relative_path": processable_file.relative_path,
                })
            
            # Create new document with standardized metadata
            standardized_doc = Document(
                page_content=doc.page_content,
                metadata=new_metadata
            )
            standardized_docs.append(standardized_doc)
        
        return standardized_docs
    
    def process_file_lazy(
        self,
        uploaded_metadata,
        max_extraction_depth: int = 3,
        loader_kwargs: Optional[Dict[str, Any]] = None
    ) -> Iterator[Document]:
        """
        Lazy version of process_file that yields documents as they're loaded.
        
        More memory efficient for large files/archives.
        
        Args:
            uploaded_metadata: UploadedFileMetadata instance
            max_extraction_depth: Maximum depth for nested archive extraction
            loader_kwargs: Additional kwargs to pass to loaders
            
        Yields:
            Document instances with standardized metadata
        """
        loader_kwargs = loader_kwargs or {}
        
        logger.info(f"Lazy processing file: {uploaded_metadata.original_filename}")
        
        try:
            with ProcessedFileSet(uploaded_metadata, self.file_type_detector) as file_set:
                
                # Extract if needed
                file_set.extract_if_archive(max_depth=max_extraction_depth)
                
                # Process each file lazily
                for processable_file in file_set.get_processable_files():
                    try:
                        # Get loader
                        loader = self.file_type_detector.create_loader(
                            processable_file.file_path,
                            processable_file.mime_type,
                            **loader_kwargs
                        )
                        
                        # Yield documents with standardized metadata
                        for doc in loader.lazy_load():
                            standardized_doc = self._standardize_single_document_metadata(
                                doc, uploaded_metadata, processable_file
                            )
                            yield standardized_doc
                            
                    except Exception as e:
                        logger.error(f"Failed to process {processable_file.file_path}: {e}")
                        # Continue with next file rather than failing entirely
                        continue
                        
        except Exception as e:
            logger.error(f"Lazy processing failed for {uploaded_metadata.original_filename}: {e}")
            raise
    
    def _standardize_single_document_metadata(
        self,
        document: Document,
        uploaded_metadata,
        processable_file: ProcessableFile
    ) -> Document:
        """Standardize metadata for a single document."""
        new_metadata = document.metadata.copy()
        
        new_metadata.update({
            "upload_id": uploaded_metadata.filename,
            "original_filename": uploaded_metadata.original_filename,
            "upload_timestamp": uploaded_metadata.upload_timestamp.isoformat(),
            "file_checksum": uploaded_metadata.checksum,
            "processing_timestamp": self._get_current_timestamp(),
            "detected_mime_type": processable_file.mime_type,
            "source_archive": processable_file.source_archive,
            "relative_path": processable_file.relative_path,
        })
        
        return Document(page_content=document.page_content, metadata=new_metadata)
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_supported_file_types(self) -> List[str]:
        """Get list of supported file types."""
        return self.file_type_detector.get_supported_extensions()