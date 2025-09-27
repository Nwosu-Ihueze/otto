#Here is where you upload files 

from datetime import datetime
import logging
import os
import shutil
from typing import Dict, List, Optional, Union
import yaml
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

from otto.utils.file_detector import FileTypeDetector
from otto.utils.misc_utils import get_default_cache_location, calculate_checksum

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {
    'csv', 'tsv', 'json','xlsx', 'xls', 'zip', 
    'tar', 'gz', 'tar.gz', 'tar.bz2', 'html', 'txt', 'md', 'rst',
    'eml', 'msg', 'pdf'
}

UPLOAD_STATUSES = {
    'PENDING': 'pending',
    'PROCESSING': 'processing', 
    'COMPLETED': 'completed',
    'FAILED': 'failed'
}


class UploadedFileMetadata:
    """Metadata for an uploaded file."""
    
    def __init__(
        self,
        filename: str,
        original_filename: str,
        file_path: str,
        file_size: int,
        mime_type: str,
        upload_timestamp: datetime,
        checksum: str,
        status: str = UPLOAD_STATUSES['COMPLETED'],
        error_message: Optional[str] = None
    ):
        self.filename = filename
        self.original_filename = original_filename
        self.file_path = file_path
        self.file_size = file_size
        self.mime_type = mime_type
        self.upload_timestamp = upload_timestamp
        self.checksum = checksum
        self.status = status
        self.error_message = error_message
    
    def to_dict(self) -> Dict:
        """Convert metadata to dictionary."""
        return {
            'filename': self.filename,
            'original_filename': self.original_filename,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'mime_type': self.mime_type,
            'upload_timestamp': self.upload_timestamp.isoformat(),
            'checksum': self.checksum,
            'status': self.status,
            'error_message': self.error_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UploadedFileMetadata':
        """Create metadata from dictionary."""
        return cls(
            filename=data['filename'],
            original_filename=data['original_filename'],
            file_path=data['file_path'],
            file_size=data['file_size'],
            mime_type=data['mime_type'],
            upload_timestamp=datetime.fromisoformat(data['upload_timestamp']),
            checksum=data['checksum'],
            status=data.get('status', UPLOAD_STATUSES['COMPLETED']),
            error_message=data.get('error_message')
        )
    

class FileUploadManager:
    """Manages file uploads for the system."""
    
    def __init__(self, upload_dir: Optional[str] = None, max_file_size: int = 500 * 1024 * 1024):  # 500MB default
        self.upload_dir = upload_dir if upload_dir else os.path.join(get_default_cache_location(), "uploads")
        self.max_file_size = max_file_size
        self.metadata_file = os.path.join(self.upload_dir, "upload_metadata.yaml")
        

        self.file_type_detector = FileTypeDetector()
        

        os.makedirs(self.upload_dir, exist_ok=True)
        

        self.file_metadata: Dict[str, UploadedFileMetadata] = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, UploadedFileMetadata]:
        """Load metadata from file."""
        if not os.path.exists(self.metadata_file):
            return {}
        
        try:
            with open(self.metadata_file, 'r') as f:
                data = yaml.safe_load(f) or {}
            return {k: UploadedFileMetadata.from_dict(v) for k, v in data.items()}
        except Exception as e:
            logger.warning(f"Failed to load upload metadata: {e}")
            return {}
    
    def _save_metadata(self) -> None:
        """Save metadata to file."""
        try:
            data = {k: v.to_dict() for k, v in self.file_metadata.items()}
            with open(self.metadata_file, 'w') as f:
                yaml.safe_dump(data, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Failed to save upload metadata: {e}")

    def _is_allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed."""
        if '.' not in filename:
            return False

        supported_extensions = self.file_type_detector.get_supported_extensions()
        
        full_ext = '.'.join(filename.split('.')[1:]).lower()
        simple_ext = filename.rsplit('.', 1)[1].lower()

        file_ext_with_dot = f".{simple_ext}"
        full_ext_with_dot = f".{full_ext}"
        
        return (file_ext_with_dot in supported_extensions or 
                full_ext_with_dot in supported_extensions or
                simple_ext in ALLOWED_EXTENSIONS or 
                full_ext in ALLOWED_EXTENSIONS)
    
    def upload_file(
        self,
        file: Union[FileStorage, str],
        original_filename: Optional[str] = None,
        description: Optional[str] = None
    ) -> UploadedFileMetadata:
        """
        Upload a file to the system.
        
        Args:
            file: FileStorage object (from web upload) or file path (string)
            original_filename: Original filename if file is a path
            description: Optional description for the upload
            
        Returns:
            UploadedFileMetadata object with upload information
        """
        try:

            if isinstance(file, str):

                if not os.path.exists(file):
                    raise FileNotFoundError(f"File not found: {file}")
                
                original_filename = original_filename or os.path.basename(file)
                file_size = os.path.getsize(file)
                

                secure_name = secure_filename(original_filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp}_{secure_name}"
                file_path = os.path.join(self.upload_dir, filename)
                
                shutil.copy2(file, file_path)

                mime_type = self.file_type_detector.detect_mime_type(file_path)
                
            else:

                if not file or not file.filename:
                    raise ValueError("No file provided")
                
                original_filename = file.filename
                file_size = 0  
                
                if not self._is_allowed_file(original_filename):
                    raise ValueError(f"File type not allowed: {original_filename}")
                

                secure_name = secure_filename(original_filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp}_{secure_name}"
                file_path = os.path.join(self.upload_dir, filename)
 
                file.save(file_path)
                file_size = os.path.getsize(file_path)
 
                provided_mime = file.mimetype if hasattr(file, 'mimetype') else None
                mime_type = self.file_type_detector.detect_mime_type(file_path, provided_mime)

            if file_size > self.max_file_size:
                os.remove(file_path) 
                raise ValueError(f"File too large: {file_size} bytes (max: {self.max_file_size} bytes)")
            

            checksum = calculate_checksum(file_path)

            metadata = UploadedFileMetadata(
                filename=filename,
                original_filename=original_filename,
                file_path=file_path,
                file_size=file_size,
                mime_type=mime_type,
                upload_timestamp=datetime.now(),
                checksum=checksum,
                status=UPLOAD_STATUSES['COMPLETED']
            )

            self.file_metadata[filename] = metadata
            self._save_metadata()
            
            logger.info(f"Successfully uploaded file: {original_filename} -> {filename}")
            return metadata
            
        except Exception as e:
            logger.error(f"Upload failed for {original_filename}: {e}")

            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
            raise
    
    def process_uploaded_file(self, filename: str, **processor_kwargs):
        """
        Process an uploaded file using the FileProcessor.
        
        Args:
            filename: Name of the uploaded file
            **processor_kwargs: Additional arguments to pass to FileProcessor
            
        Returns:
            ProcessingResult with documents and processing information
        """
        from otto.utils.documents.file_processor import FileProcessor
        
        metadata = self.get_file_metadata(filename)
        if not metadata:
            raise ValueError(f"File not found: {filename}")

        metadata.status = UPLOAD_STATUSES['PROCESSING']
        metadata.error_message = None
        self._save_metadata()
        
        try:

            processor = FileProcessor(self.file_type_detector)
            result = processor.process_file(metadata, **processor_kwargs)

            if result.status.value == 'completed':
                metadata.status = UPLOAD_STATUSES['COMPLETED']
                metadata.error_message = None
            else:
                metadata.status = UPLOAD_STATUSES['FAILED']
                metadata.error_message = "; ".join(result.errors) if result.errors else "Processing failed"
            
            self._save_metadata()
            
            logger.info(f"Processing completed for {filename}: {len(result.documents)} documents extracted")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process uploaded file {filename}: {e}")
            metadata.status = UPLOAD_STATUSES['FAILED']
            metadata.error_message = str(e)
            self._save_metadata()
            raise
    
    def process_uploaded_file_lazy(self, filename: str, **processor_kwargs):
        """
        Lazy process an uploaded file using the FileProcessor.
        
        Args:
            filename: Name of the uploaded file
            **processor_kwargs: Additional arguments to pass to FileProcessor
            
        Yields:
            Document instances as they're processed
        """
        from otto.utils.documents.file_processor import FileProcessor
        
        metadata = self.get_file_metadata(filename)
        if not metadata:
            raise ValueError(f"File not found: {filename}")
        
        metadata.status = UPLOAD_STATUSES['PROCESSING']
        metadata.error_message = None
        self._save_metadata()
        
        try:

            processor = FileProcessor(self.file_type_detector)
            
            document_count = 0
            for document in processor.process_file_lazy(metadata, **processor_kwargs):
                document_count += 1
                yield document
 
            metadata.status = UPLOAD_STATUSES['COMPLETED']
            metadata.error_message = None
            self._save_metadata()
            
            logger.info(f"Lazy processing completed for {filename}: {document_count} documents processed")
            
        except Exception as e:
            logger.error(f"Failed to lazy process uploaded file {filename}: {e}")
            metadata.status = UPLOAD_STATUSES['FAILED']
            metadata.error_message = str(e)
            self._save_metadata()
            raise
    
    def get_file_metadata(self, filename: str) -> Optional[UploadedFileMetadata]:
        """Get metadata for an uploaded file."""
        return self.file_metadata.get(filename)
    
    def list_uploaded_files(self) -> List[UploadedFileMetadata]:
        """List all uploaded files."""
        return list(self.file_metadata.values())
    
    def delete_uploaded_file(self, filename: str) -> bool:
        """Delete an uploaded file and its metadata."""
        if filename not in self.file_metadata:
            return False
        
        metadata = self.file_metadata[filename]
        

        if os.path.exists(metadata.file_path):
            os.remove(metadata.file_path)
        

        del self.file_metadata[filename]
        self._save_metadata()
        
        logger.info(f"Deleted uploaded file: {filename}")
        return True
    
    def get_processing_status(self, filename: str) -> Optional[str]:
        """Get the processing status of an uploaded file."""
        metadata = self.get_file_metadata(filename)
        return metadata.status if metadata else None
    
    def get_supported_file_types(self) -> List[str]:
        """Get list of supported file extensions."""
        return self.file_type_detector.get_supported_extensions()
    