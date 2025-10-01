import logging
import os
import argparse
import sys
from typing import Optional
from pathlib import Path

from otto.config_loader import OTTOConfig, load_config, merge_config_with_args
from otto.preprocess import DocumentProcessor
from otto.trainer import SLMTrainer
from otto.upload_file import FileUploadManager


logger = logging.getLogger(__name__)

def complete_pipeline_cli(
    file_path: str,
    config: OTTOConfig,
    stream_processing: bool = True,
) -> None:
    """
    Complete pipeline: Upload -> Process -> Preprocess -> Train
    
    Args:
        file_path: Path to file to upload and train on
        config: OTTOConfig instance with all settings
        stream_processing: If True, stream documents to avoid memory issues
    """
    print("=" * 60)
    print("OTTO SLM COMPLETE PIPELINE")
    print("=" * 60)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Step 1: Upload and Process
    print("\nSTEP 1: UPLOADING AND PROCESSING FILE")
    print("-" * 40)
    
    manager = FileUploadManager(
        upload_dir=config.paths.upload_dir,
        max_file_size=config.system.max_file_size
    )
    
    try:
        print(f"Uploading: {file_path}")
        metadata = manager.upload_file(file_path)
        print(f"  ✓ Uploaded as: {metadata.filename}")
        print(f"  ✓ File size: {metadata.file_size:,} bytes")
        print(f"  ✓ MIME type: {metadata.mime_type}")
        

        if stream_processing:
            print(f"Processing uploaded file (streaming mode)...")
            documents = []
            doc_count = 0
            
            for doc in manager.process_uploaded_file_lazy(metadata.filename):
                documents.append(doc)
                doc_count += 1
                if doc_count % 100000 == 0:
                    print(f"  Loaded {doc_count:,} documents...")
            
            print(f"  ✓ Processing completed!")
            print(f"  ✓ Documents extracted: {len(documents)}")
            
            if not documents:
                print("No documents extracted. Cannot proceed with training.")
                return
        else:
            print(f"Processing uploaded file...")
            result = manager.process_uploaded_file(metadata.filename)
            
            if result.status.value == 'completed':
                print(f"  ✓ Processing completed!")
                print(f"  ✓ Documents extracted: {len(result.documents)}")
                print(f"  ✓ Files processed: {result.processed_files_count}")
                
                if not result.documents:
                    print("No documents extracted. Cannot proceed with training.")
                    return
                
                documents = result.documents
            else:
                print(f"Processing failed: {'; '.join(result.errors)}")
                return
            
    except Exception as e:
        print(f"Upload/Processing failed: {e}")
        raise
    
    # Step 2: Preprocessing for Training
    print(f"\nSTEP 2: PREPROCESSING FOR TRAINING")
    print("-" * 40)
    
    try:
        doc_processor = DocumentProcessor(
            output_dir=config.paths.training_data_dir,
            tokenizer_name=config.data.tokenizer,
            train_split=config.data.train_split,
            min_doc_length=config.data.min_doc_length,
            max_doc_length=config.data.max_doc_length
        )
        
        print(f"Preprocessing {len(documents)} documents...")
        preprocessing_stats = doc_processor.process_documents(
            documents=documents,
            generate_binary=True,
            save_text=False #Switch to True if you want to save text files
        )

        del documents
        
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
        
        verification = doc_processor.verify_binary_files()
        if not all(verification.values()):
            print(f"Binary file verification failed: {verification}")
            return
        
        print(f"  ✓ Binary files verified")
        
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        raise

    # Step 3: Training
    print(f"\nSTEP 3: TRAINING SMALL LANGUAGE MODEL")
    print("-" * 40)
    
    try:
        train_tokens = preprocessing_stats['token_stats']['train']['total_tokens']
        min_tokens_needed = config.training.max_iters * config.training.batch_size * config.model.block_size
        
        if train_tokens < min_tokens_needed:
            print(f"  Warning: Limited training data")
            print(f"   Available tokens: {train_tokens:,}")
            print(f"   Recommended: {min_tokens_needed:,}")
            print(f"   Reducing training iterations...")
            config.training.max_iters = max(100, train_tokens // (config.training.batch_size * config.model.block_size))
        
        trainer = SLMTrainer(
            training_data_dir=config.paths.training_data_dir,
            output_dir=config.paths.model_output_dir,
            learning_rate=config.training.learning_rate,
            max_iters=config.training.max_iters,
            warmup_steps=config.training.warmup_steps,
            min_lr=config.training.min_lr,
            eval_iters=config.training.eval_iters,
            batch_size=config.training.batch_size,
            block_size=config.model.block_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            n_layer=config.model.n_layer,
            n_head=config.model.n_head,
            n_embd=config.model.n_embd,
            dropout=config.model.dropout,
            device=config.system.device,
            dtype=config.system.dtype
        )
        
        print(f"Starting training...")
        print(f"  Model: {config.model.n_layer} layers, {config.model.n_head} heads, {config.model.n_embd} dim")
        print(f"  Iterations: {config.training.max_iters}")
        print(f"  Batch size: {config.training.batch_size}")
        print(f"  Context size: {config.model.block_size}")
        print(f"  Device: {trainer.device}")
        
        training_results = trainer.train(save_every=max(500, config.training.max_iters//4))
        
        print(f"  ✓ Training completed!")
        print(f"  ✓ Final train loss: {training_results['final_train_loss']:.4f}")
        print(f"  ✓ Final val loss: {training_results['final_val_loss']:.4f}")
        print(f"  ✓ Best val loss: {training_results['best_val_loss']:.4f}")
        print(f"  ✓ Model saved to: {training_results['best_model_path']}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        raise
    
    # Step 4: Test Generation
    print(f"\nSTEP 4: TESTING TEXT GENERATION")
    print("-" * 40)
    print("Enter prompts to test the model. Type 'quit' or 'exit' to finish.")
    print("Commands: 'quit', 'exit', or press Ctrl+C\n")
    
    try:
        while True:
            try:
                prompt = input("Enter prompt (or 'quit' to exit): ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not prompt:
                    print("Empty prompt. Please enter something.\n")
                    continue
                
                print(f"Generating...")
                generated = trainer.generate_text(
                    prompt=prompt, 
                    max_new_tokens=100, 
                    temperature=0.6,
                    top_k=40
                )
                print(f"\n{generated}\n")
                
            except KeyboardInterrupt:
                print("\n\nExiting generation mode...")
                break
            except EOFError:
                break
                
    except Exception as e:
        print(f"Text generation failed: {e}")
    
    print(f"\nPIPELINE COMPLETED SUCCESSFULLY!")
    print(f"   Training data: {config.paths.training_data_dir}")
    print(f"   Model outputs: {config.paths.model_output_dir}")
    print("=" * 60)


def cli():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Complete SLM pipeline: Upload → Process → Preprocess → Train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("file_path", help="Path to the file to upload and train on")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--no-stream", action="store_true", 
                       help="Disable streaming mode (loads all docs at once, may cause OOM)")
    
    parser.add_argument("--max-iters", type=int, help="Override max training iterations")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--n-layer", type=int, help="Override number of layers")
    parser.add_argument("--n-embd", type=int, help="Override embedding dimension")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config = load_config(args.config)
    config = merge_config_with_args(
        config,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        n_layer=args.n_layer,
        n_embd=args.n_embd
    )
    
    complete_pipeline_cli(args.file_path, config, stream_processing=not args.no_stream)

if __name__ == "__main__":
    cli()