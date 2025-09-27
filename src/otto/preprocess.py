import os
import numpy as np
import tiktoken
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from tqdm.auto import tqdm
import logging
import re

from otto.utils.documents.document_utils import Document

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Processes documents from upload pipeline into training-ready format for SLM.
    
    Handles text consolidation, cleaning, tokenization, and binary file generation
    for efficient training data loading.
    """
    
    def __init__(
        self,
        output_dir: str = "training_data",
        tokenizer_name: str = "gpt2",
        train_split: float = 0.9,
        min_doc_length: int = 10,
        max_doc_length: Optional[int] = None,
        chunk_size: int = 1024,
        dtype: str = "uint16"
    ):
        """
        Initialize the document processor.
        
        Args:
            output_dir: Directory to save processed files
            tokenizer_name: Tokenizer to use (gpt2, gpt2-medium, etc.)
            train_split: Fraction of data for training (rest goes to validation)
            min_doc_length: Minimum document length in characters
            max_doc_length: Maximum document length in characters (None = no limit)
            chunk_size: Size of chunks for processing large datasets
            dtype: Data type for binary files (uint16 for vocab < 65536)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.train_split = train_split
        self.min_doc_length = min_doc_length
        self.max_doc_length = max_doc_length
        self.chunk_size = chunk_size
        self.dtype = getattr(np, dtype)
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding(tokenizer_name)
            self.vocab_size = self.tokenizer.max_token_value + 1
            logger.info(f"Initialized tokenizer: {tokenizer_name}, vocab_size: {self.vocab_size}")
        except Exception as e:
            raise ValueError(f"Failed to initialize tokenizer {tokenizer_name}: {e}")
        
        # Validate dtype can handle vocab size
        max_value = np.iinfo(self.dtype).max
        if self.vocab_size > max_value:
            raise ValueError(f"Vocab size {self.vocab_size} exceeds {dtype} max value {max_value}")
    
    def process_documents(
        self, 
        documents: List[Document],
        generate_binary: bool = True,
        save_text: bool = False
    ) -> Dict[str, any]:
        """
        Process documents into training-ready format.
        
        Args:
            documents: List of Document objects from upload pipeline
            generate_binary: Whether to generate binary training files
            save_text: Whether to save consolidated text files
            
        Returns:
            Dictionary with processing statistics and file paths
        """
        logger.info(f"Processing {len(documents)} documents...")
        
    
        cleaned_texts = self._clean_and_filter_documents(documents)

        train_texts, val_texts = self._split_data(cleaned_texts)

        text_files = {}
        if save_text:
            text_files = self._save_text_files(train_texts, val_texts)
        

        binary_files = {}
        token_stats = {}
        if generate_binary:
            binary_files, token_stats = self._generate_binary_files(train_texts, val_texts)

        stats = {
            "total_documents": len(documents),
            "processed_documents": len(cleaned_texts),
            "train_documents": len(train_texts),
            "val_documents": len(val_texts),
            "text_files": text_files,
            "binary_files": binary_files,
            "token_stats": token_stats,
            "vocab_size": self.vocab_size,
            "output_dir": str(self.output_dir)
        }
        
        self._save_stats(stats)
        logger.info("Document processing completed!")
        return stats
    
    def _clean_and_filter_documents(self, documents: List[Document]) -> List[str]:
        """Clean and filter documents based on length and quality."""
        cleaned_texts = []
        
        for doc in tqdm(documents, desc="Cleaning documents"):
            text = doc.page_content
            
 
            text = self._clean_text(text)
            
 
            if len(text) < self.min_doc_length:
                continue
            
            if self.max_doc_length and len(text) > self.max_doc_length:
            
                text = text[:self.max_doc_length]
            
            cleaned_texts.append(text)
        
        logger.info(f"Filtered {len(documents)} → {len(cleaned_texts)} documents")
        return cleaned_texts
    
    def _clean_text(self, text: str) -> str:
        """Apply text cleaning operations."""

        text = re.sub(r'\s+', ' ', text)
        
 
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
 
        text = text.encode('utf-8', errors='ignore').decode('utf-8')

        text = text.strip()
        
        return text
    
    def _split_data(self, texts: List[str]) -> Tuple[List[str], List[str]]:
        """Split data into train and validation sets."""

        import random
        shuffled_texts = texts.copy()
        random.shuffle(shuffled_texts)
        
        split_idx = int(len(shuffled_texts) * self.train_split)
        train_texts = shuffled_texts[:split_idx]
        val_texts = shuffled_texts[split_idx:]
        
        logger.info(f"Split data: {len(train_texts)} train, {len(val_texts)} validation")
        return train_texts, val_texts
    
    def _save_text_files(self, train_texts: List[str], val_texts: List[str]) -> Dict[str, str]:
        """Save consolidated text files."""
        files = {}
        

        train_file = self.output_dir / "train.txt"
        with open(train_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(train_texts))
        files['train_txt'] = str(train_file)
        

        val_file = self.output_dir / "val.txt"
        with open(val_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(val_texts))
        files['val_txt'] = str(val_file)
        
        logger.info(f"Saved text files: {train_file}, {val_file}")
        return files
    
    def _generate_binary_files(
        self, 
        train_texts: List[str], 
        val_texts: List[str]
    ) -> Tuple[Dict[str, str], Dict[str, any]]:
        """Generate binary training files."""
        files = {}
        stats = {}
        

        if train_texts:
            train_file, train_stats = self._tokenize_and_save(train_texts, "train")
            files['train_bin'] = train_file
            stats['train'] = train_stats
        

        if val_texts:
            val_file, val_stats = self._tokenize_and_save(val_texts, "val")
            files['val_bin'] = val_file
            stats['val'] = val_stats
        
        return files, stats
    
    def _tokenize_and_save(self, texts: List[str], split: str) -> Tuple[str, Dict[str, any]]:
        """Tokenize texts and save as binary file."""
        filename = self.output_dir / f"{split}.bin"
        

        total_tokens = 0
        doc_lengths = []
        
        for text in tqdm(texts, desc=f"Calculating tokens for {split}"):
            tokens = self.tokenizer.encode_ordinary(text)
            doc_lengths.append(len(tokens))
            total_tokens += len(tokens)
        
        logger.info(f"{split}: {total_tokens:,} total tokens from {len(texts)} documents")
        

        arr = np.memmap(filename, dtype=self.dtype, mode='w+', shape=(total_tokens,))
      
        idx = 0
        for i, text in enumerate(tqdm(texts, desc=f"Writing {split}.bin")):
            tokens = self.tokenizer.encode_ordinary(text)
            token_array = np.array(tokens, dtype=self.dtype)
            

            arr[idx:idx + len(token_array)] = token_array
            idx += len(token_array)

        arr.flush()
        del arr  
        

        stats = {
            "total_tokens": total_tokens,
            "num_documents": len(texts),
            "avg_tokens_per_doc": total_tokens / len(texts) if texts else 0,
            "min_tokens": min(doc_lengths) if doc_lengths else 0,
            "max_tokens": max(doc_lengths) if doc_lengths else 0,
            "file_size_mb": os.path.getsize(filename) / 1024 / 1024
        }
        
        logger.info(f"Saved {filename}: {stats['file_size_mb']:.1f}MB")
        return str(filename), stats
    
    def _save_stats(self, stats: Dict[str, any]) -> None:
        """Save processing statistics."""
        import json
        
        stats_file = self.output_dir / "processing_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Saved statistics to {stats_file}")
    
    def get_batch_function(self, block_size: int, batch_size: int, device: str = 'cpu'):
        """
        Generate a get_batch function for training.
        
        Args:
            block_size: Context window size
            batch_size: Batch size for training
            device: Device to load data to
            
        Returns:
            Function that returns training batches
        """
        def get_batch(split: str):
            """Get a batch of training data."""
            import torch

            if split == 'train':
                data_file = self.output_dir / "train.bin"
            else:
                data_file = self.output_dir / "val.bin"
            
            if not data_file.exists():
                raise FileNotFoundError(f"Binary file not found: {data_file}")
            

            data = np.memmap(data_file, dtype=self.dtype, mode='r')
            
            # Generate random starting positions
            ix = torch.randint(len(data) - block_size, (batch_size,))
            

            x = torch.stack([
                torch.from_numpy(data[i:i+block_size].astype(np.int64)) 
                for i in ix
            ])
            y = torch.stack([
                torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) 
                for i in ix
            ])

            if device == 'cuda':
                x = x.pin_memory().to(device, non_blocking=True)
                y = y.pin_memory().to(device, non_blocking=True)
            else:
                x = x.to(device)
                y = y.to(device)
            
            return x, y
        
        return get_batch
    
    def verify_binary_files(self) -> Dict[str, bool]:
        """Verify that binary files were created correctly."""
        results = {}
        
        for split in ['train', 'val']:
            file_path = self.output_dir / f"{split}.bin"
            
            if not file_path.exists():
                results[split] = False
                continue
            
            try:

                data = np.memmap(file_path, dtype=self.dtype, mode='r')
                
                if len(data) > 100:
                    sample_tokens = data[:100]

                    if np.all(sample_tokens < self.vocab_size):
                        results[split] = True
                    else:
                        results[split] = False
                        logger.error(f"{split}.bin contains invalid tokens > vocab_size")
                else:
                    results[split] = False
                    logger.error(f"{split}.bin is too small")
                    
            except Exception as e:
                results[split] = False
                logger.error(f"Error verifying {split}.bin: {e}")
        
        return results