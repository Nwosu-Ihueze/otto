# OTTO - Small Language Model Training Pipeline

A complete end-to-end pipeline for training specialized Small Language Models (SLMs) on custom business data. OTTO enables organizations to train domain-specific language models without relying on expensive LLM fine-tuning or external APIs.

## Overview

OTTO provides a streamlined workflow: **Upload → Process → Preprocess → Train → Evaluate**

The pipeline automatically handles file processing, data cleaning, tokenization, model training, and evaluation to produce specialized language models tailored to your specific use case.

## Current Status: First Iteration

**Works Best With:**
- Call transcripts and customer conversations
- Text-based business documents
- Natural language content (emails, reports, reviews)
- Conversational data and dialog systems

**Limited Support For:**
- Structured data (CSV, spreadsheets) - produces incoherent output
- Mixed media content requiring vision/audio processing
- Highly technical formats requiring specialized preprocessing

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd otto

# Install dependencies
uv install

# Install system dependencies (optional, for better file type detection)
brew install libmagic  # macOS
# sudo apt-get install libmagic1  # Ubuntu
```

### Basic Usage

```bash
# Train a model on your data
uv run src/otto/cli_runner.py your_data.txt

# Custom training parameters
uv run src/otto/cli_runner.py your_data.txt \
  --max-iters 2000 \
  --batch-size 32 \
  --block-size 128

# Skip training, only preprocess
uv run src/otto/cli_runner.py your_data.txt --no-training

# Run inference and evaluation
uv run src/otto/inference.py model_outputs/best_model.pt --interactive
```

## Architecture

### Core Components

1. **FileUploadManager** - Handles file uploads and basic validation
2. **FileTypeDetector** - Intelligent file type detection with fallback chains
3. **ProcessedFileSet** - Manages archive extraction and temporary file handling
4. **FileProcessor** - Coordinates document processing pipeline
5. **DocumentProcessor** - Converts documents to training-ready format
6. **SLMTrainer** - Trains GPT-style transformer models
7. **SLMInference** - Handles model loading and text generation

### Data Flow

```
Raw Files → Upload → Type Detection → Archive Extraction → Document Loading → 
Text Cleaning → Tokenization → Binary Files → Model Training → Evaluation
```

## Supported File Types

- **Text**: .txt, .md, .rst
- **Structured**: .csv, .tsv, .json, .jsonl
- **Archives**: .zip, .tar, .tar.gz, .tar.bz2, .gz

## Training Configuration

### Model Architecture
- GPT-style transformer with causal attention
- Configurable layers, heads, and embedding dimensions
- Support for mixed precision training (FP16/BF16)
- Automatic vocabulary size detection

### Default Settings
```python
# Small model for testing
n_layer=4, n_head=4, n_embd=256, block_size=64

# Production model
n_layer=6, n_head=6, n_embd=384, block_size=128
```

### Training Features
- Gradient accumulation for large effective batch sizes
- Learning rate warmup and cosine decay
- Automatic checkpointing and best model saving
- Memory-efficient data loading with memory mapping
- Progress tracking and loss monitoring

## Example Use Cases

### Customer Service Chatbot
```bash
# Train on call transcripts
uv run src/otto/cli_runner.py customer_calls.txt \
  --max-iters 5000 \
  --batch-size 32
```

### Domain-Specific Text Generation
```bash
# Train on legal documents
uv run src/otto/cli_runner.py legal_corpus.txt \
  --block-size 256 \  # Longer context for complex documents
  --n-layer 8
```

### Interactive Testing
```bash
# Test your trained model
uv run src/otto/inference.py model_outputs/best_model.pt --interactive

# Evaluate model performance
uv run src/otto/inference.py model_outputs/best_model.pt \
  --evaluate --test-data training_data/val.bin
```

## Performance Expectations

### Good Results Expected
- **Perplexity**: 10-100 range
- **Generation**: Coherent, domain-relevant text
- **Training Loss**: Decreases from ~10 to 2-4 range

### Warning Signs
- **Perplexity**: >1000 indicates poor learning
- **Generation**: Random token sequences
- **Loss**: Not decreasing or very high (>8)

## Limitations (Current Version)

### Data Requirements
- Minimum 100k+ tokens for meaningful training
- Text should be naturally flowing (not structured tables)
- Works best with conversational or narrative content

### Technical Limitations
- CPU training only (GPU support planned)
- No distributed training
- Limited to text-only data
- No fine-tuning from pretrained models

### Known Issues
- CSV data produces incoherent output without preprocessing
- Very small datasets lead to overfitting
- No support for multi-modal data

## TODO: Planned Improvements

### High Priority

#### 1. Structured Data Support
- [ ] CSV-to-natural-language converter
- [ ] Template-based data description generation
- [ ] Configurable data formatting strategies
- [ ] Support for tabular data relationships

#### 2. GPU Training Support
- [ ] CUDA acceleration for training
- [ ] Multi-GPU support for larger models
- [ ] Memory optimization for large datasets
- [ ] Automatic device detection and optimization

#### 3. Enhanced Data Processing
- [ ] Domain-specific text cleaning
- [ ] Advanced tokenization strategies
- [ ] Support for code and technical documentation
- [ ] Multi-language text handling

### Medium Priority

#### 4. Model Architecture Improvements
- [ ] Support for different transformer variants
- [ ] Configurable attention mechanisms
- [ ] Model size recommendations based on data
- [ ] Pretrained model fine-tuning capabilities

#### 5. Advanced Training Features
- [ ] Distributed training across multiple machines
- [ ] Curriculum learning strategies
- [ ] Advanced optimization techniques
- [ ] Hyperparameter auto-tuning

#### 6. Business-Specific Features
- [ ] Task-specific model heads (classification, sentiment)
- [ ] Domain adaptation techniques
- [ ] Privacy-preserving training options
- [ ] Model compression and quantization

### Low Priority

#### 7. User Experience
- [ ] Web-based training interface
- [ ] Real-time training monitoring
- [ ] Model comparison tools
- [ ] Automated report generation

#### 8. Integration & Deployment
- [ ] API server for model serving
- [ ] Docker containerization
- [ ] Cloud deployment templates
- [ ] Integration with business tools

#### 9. Evaluation & Monitoring
- [ ] Domain-specific evaluation metrics
- [ ] A/B testing framework
- [ ] Model drift detection
- [ ] Performance monitoring dashboard

## Contributing

### Development Setup
```bash
# Install development dependencies
uv install --dev

# Run tests
pytest tests/

# Format code
black src/
isort src/
```

### Adding New File Types
1. Create loader in `src/otto/data_loaders/`
2. Add MIME type mapping in `FileTypeDetector`
3. Add tests for the new loader
4. Update documentation

### Adding New Model Architectures
1. Implement model in `src/otto/models/`
2. Update `SLMTrainer` to support new architecture
3. Add configuration validation
4. Test training and inference

## License

MIT License

## Citation

If you use OTTO in your research or business applications, please cite:

```
@software{otto_slm_pipeline,
  title={OTTO: Small Language Model Training Pipeline},
  author={[Rosemary Nwosu-Ihueze]},
  year={2025},
  url={https://github.com/nwosu-ihueze/otto}
}
```

## Support

- **Issues**: Report bugs and feature requests via GitHub Issues
- **Documentation**: See `docs/` directory for detailed guides
- **Community**: [Add community links when available]

---

**Note**: This is the first iteration of OTTO. While functional for text-based training, significant improvements are planned for structured data support and production deployment features.