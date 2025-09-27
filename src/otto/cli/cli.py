import logging
import os
import argparse
import sys
from typing import Optional
from pathlib import Path

from otto.preprocess import DocumentProcessor
from otto.trainer import SLMTrainer
from otto.upload_file import FileUploadManager


logger = logging.getLogger(__name__)

def complete_pipeline_cli(
    file_path: str,
    upload_dir: Optional[str] = None,
    training_data_dir: str = "training_data",
    model_output_dir: str = "model_outputs",
    max_size: Optional[int] = None,
    # Preprocessing options
    train_split: float = 0.9,
    min_doc_length: int = 50,
    max_doc_length: Optional[int] = None,
    # Training options
    do_training: bool = True,
    max_iters: int = 1000,  # Reduced for testing
    batch_size: int = 16,   # Reduced for testing
    block_size: int = 64,   # Reduced for testing
    eval_iters: int = 100,  # Reduced for testing
    # Model options
    n_layer: int = 4,       # Smaller model for testing
    n_head: int = 4,
    n_embd: int = 256,
    dropout: float = 0.1,
) -> None:
    """
    Complete pipeline: Upload -> Process -> Preprocess -> Train
    
    Args:
        file_path: Path to file to upload and train on
        upload_dir: Directory to upload files to
        training_data_dir: Directory to save preprocessed training data
        model_output_dir: Directory to save trained model
        max_size: Maximum file size in bytes
        train_split: Fraction of data for training
        min_doc_length: Minimum document length in characters
        max_doc_length: Maximum document length in characters
        do_training: Whether to run training after preprocessing
        max_iters: Number of training iterations
        batch_size: Training batch size
        block_size: Context window size
        eval_iters: Evaluation frequency
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension
        dropout: Dropout rate
    """
    print("=" * 60)
    print("OTTO SLM COMPLETE PIPELINE")
    print("=" * 60)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Step 1: Upload and Process
    print("\nSTEP 1: UPLOADING AND PROCESSING FILE")
    print("-" * 40)
    
    max_file_size = max_size if max_size else 500 * 1024 * 1024
    manager = FileUploadManager(upload_dir=upload_dir, max_file_size=max_file_size)
    
    try:
        # Upload file
        print(f"Uploading: {file_path}")
        metadata = manager.upload_file(file_path)
        print(f"  ✓ Uploaded as: {metadata.filename}")
        print(f"  ✓ File size: {metadata.file_size:,} bytes")
        print(f"  ✓ MIME type: {metadata.mime_type}")
        
        # Process file
        print(f"Processing uploaded file...")
        result = manager.process_uploaded_file(metadata.filename)
        
        if result.status.value == 'completed':
            print(f"  ✓ Processing completed!")
            print(f"  ✓ Documents extracted: {len(result.documents)}")
            print(f"  ✓ Files processed: {result.processed_files_count}")
            
            if not result.documents:
                print("No documents extracted. Cannot proceed with training.")
                return
        else:
            print(f"Processing failed: {'; '.join(result.errors)}")
            return
            
    except Exception as e:
        print(f"Upload/Processing failed: {e}")
        raise
    
    # Step 2: Preprocessing for Training
    print(f"\n STEP 2: PREPROCESSING FOR TRAINING")
    print("-" * 40)
    
    try:
        doc_processor = DocumentProcessor(
            output_dir=training_data_dir,
            train_split=train_split,
            min_doc_length=min_doc_length,
            max_doc_length=max_doc_length
        )
        
        print(f"Preprocessing {len(result.documents)} documents...")
        preprocessing_stats = doc_processor.process_documents(
            documents=result.documents,
            generate_binary=True,
            save_text=True
        )
        
        print(f"  ✓ Preprocessing completed!")
        print(f"  ✓ Total documents: {preprocessing_stats['total_documents']}")
        print(f"  ✓ Processed documents: {preprocessing_stats['processed_documents']}")
        print(f"  ✓ Training documents: {preprocessing_stats['train_documents']}")
        print(f"  ✓ Validation documents: {preprocessing_stats['val_documents']}")
        print(f"  ✓ Vocab size: {preprocessing_stats['vocab_size']}")
        
        if 'token_stats' in preprocessing_stats:
            train_stats = preprocessing_stats['token_stats'].get('train', {})
            if train_stats:
                print(f"  ✓ Training tokens: {train_stats['total_tokens']:,}")
                print(f"  ✓ Avg tokens per doc: {train_stats['avg_tokens_per_doc']:.1f}")
        
        # Verify binary files
        verification = doc_processor.verify_binary_files()
        if not all(verification.values()):
            print(f"Binary file verification failed: {verification}")
            return
        
        print(f"  ✓ Binary files verified")
        
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        raise
    
    # Step 3: Training (optional)
    if not do_training:
        print(f"\n SKIPPING TRAINING (do_training=False)")
        print(f"Training data is ready in: {training_data_dir}")
        return
    
    print(f"\n STEP 3: TRAINING SMALL LANGUAGE MODEL")
    print("-" * 40)
    
    try:
        # Check if we have enough data for training
        train_tokens = preprocessing_stats['token_stats']['train']['total_tokens']
        min_tokens_needed = max_iters * batch_size * block_size
        
        if train_tokens < min_tokens_needed:
            print(f"⚠️  Warning: Limited training data")
            print(f"   Available tokens: {train_tokens:,}")
            print(f"   Recommended: {min_tokens_needed:,}")
            print(f"   Reducing training iterations...")
            max_iters = min(max_iters, train_tokens // (batch_size * block_size))
            max_iters = max(100, max_iters)  # Minimum 100 iterations
        
        trainer = SLMTrainer(
            training_data_dir=training_data_dir,
            output_dir=model_output_dir,
            max_iters=max_iters,
            batch_size=batch_size,
            block_size=block_size,
            eval_iters=eval_iters,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout
        )
        
        print(f"Starting training...")
        print(f"  Model: {trainer.model_config.n_layer} layers, {trainer.model_config.n_head} heads, {trainer.model_config.n_embd} dim")
        print(f"  Iterations: {max_iters}")
        print(f"  Batch size: {batch_size}")
        print(f"  Context size: {block_size}")
        print(f"  Device: {trainer.device}")
        
        training_results = trainer.train(save_every=max(500, max_iters//4))
        
        print(f"  ✓ Training completed!")
        print(f"  ✓ Final train loss: {training_results['final_train_loss']:.4f}")
        print(f"  ✓ Final val loss: {training_results['final_val_loss']:.4f}")
        print(f"  ✓ Best val loss: {training_results['best_val_loss']:.4f}")
        print(f"  ✓ Model saved to: {training_results['best_model_path']}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        raise
    
    # Step 4: Test Generation
    print(f"\n STEP 4: TESTING TEXT GENERATION")
    print("-" * 40)
    
    try:
        # Generate some sample text
        sample_prompts = ["The", "Once upon a time", "In the beginning"]
        
        for prompt in sample_prompts:
            print(f"\nPrompt: '{prompt}'")
            generated = trainer.generate_text(
                prompt=prompt, 
                max_new_tokens=50, 
                temperature=0.8,
                top_k=40
            )
            print(f"Generated: {generated}")
            
    except Exception as e:
        print(f"Text generation failed: {e}")
    
    print(f"\n PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"   Training data: {training_data_dir}")
    print(f"   Model outputs: {model_output_dir}")
    print("=" * 60)


def cli():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Complete SLM pipeline: Upload → Process → Preprocess → Train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "file_path",
        help="Path to the file to upload and train on"
    )
    
    # File handling
    parser.add_argument(
        "--upload-dir", 
        help="Directory to upload files to"
    )
    parser.add_argument(
        "--training-data-dir", 
        default="training_data",
        help="Directory to save preprocessed training data"
    )
    parser.add_argument(
        "--model-output-dir", 
        default="model_outputs",
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--max-size", 
        type=int,
        help="Maximum file size in bytes"
    )
    
    # Preprocessing options
    parser.add_argument(
        "--train-split", 
        type=float, 
        default=0.9,
        help="Fraction of data for training (rest for validation)"
    )
    parser.add_argument(
        "--min-doc-length", 
        type=int, 
        default=50,
        help="Minimum document length in characters"
    )
    parser.add_argument(
        "--max-doc-length", 
        type=int,
        help="Maximum document length in characters"
    )
    
    # Training options
    parser.add_argument(
        "--no-training", 
        action="store_true",
        help="Skip training step (only preprocess)"
    )
    parser.add_argument(
        "--max-iters", 
        type=int, 
        default=1000,
        help="Number of training iterations"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=16,
        help="Training batch size"
    )
    parser.add_argument(
        "--block-size", 
        type=int, 
        default=64,
        help="Context window size"
    )
    parser.add_argument(
        "--eval-iters", 
        type=int, 
        default=100,
        help="Evaluation frequency"
    )
    
    # Model options
    parser.add_argument(
        "--n-layer", 
        type=int, 
        default=4,
        help="Number of transformer layers"
    )
    parser.add_argument(
        "--n-head", 
        type=int, 
        default=4,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--n-embd", 
        type=int, 
        default=256,
        help="Embedding dimension"
    )
    parser.add_argument(
        "--dropout", 
        type=float, 
        default=0.1,
        help="Dropout rate"
    )
    
    # Logging
    parser.add_argument(
        "--log-level", 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run pipeline
    complete_pipeline_cli(
        file_path=args.file_path,
        upload_dir=args.upload_dir,
        training_data_dir=args.training_data_dir,
        model_output_dir=args.model_output_dir,
        max_size=args.max_size,
        train_split=args.train_split,
        min_doc_length=args.min_doc_length,
        max_doc_length=args.max_doc_length,
        do_training=not args.no_training,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        block_size=args.block_size,
        eval_iters=args.eval_iters,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout
    )


if __name__ == "__main__":
    cli()