import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True
    block_size: int = 128
    vocab_size: int = 50257


@dataclass
class TrainingConfig:
    """Training parameters configuration."""
    max_iters: int = 2000
    batch_size: int = 16
    learning_rate: float = 0.0001
    warmup_steps: int = 1000
    min_lr: float = 0.00005
    eval_iters: int = 500
    gradient_accumulation_steps: int = 32


@dataclass
class DataConfig:
    """Data processing configuration."""
    train_split: float = 0.9
    min_doc_length: int = 50
    max_doc_length: Optional[int] = None
    tokenizer: str = "gpt2"


@dataclass
class PathsConfig:
    """File paths configuration."""
    upload_dir: Optional[str] = None
    training_data_dir: str = "training_data"
    model_output_dir: str = "model_outputs"


@dataclass
class SystemConfig:
    """System configuration."""
    max_file_size: int = 524288000  # 500MB
    device: Optional[str] = None
    dtype: Optional[str] = None


@dataclass
class OTTOConfig:
    """Complete OTTO configuration."""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    paths: PathsConfig
    system: SystemConfig
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.model.n_embd % self.model.n_head != 0:
            raise ValueError(
                f"n_embd ({self.model.n_embd}) must be divisible by n_head ({self.model.n_head})"
            )
        
        if self.model.n_layer <= 0:
            raise ValueError(f"n_layer must be positive, got {self.model.n_layer}")
        
        if self.model.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.model.block_size}")
        
        if self.training.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.training.batch_size}")
        
        if self.training.max_iters <= 0:
            raise ValueError(f"max_iters must be positive, got {self.training.max_iters}")
        
        if not 0 < self.training.learning_rate < 1:
            raise ValueError(f"learning_rate must be between 0 and 1, got {self.training.learning_rate}")
        
        if not 0 < self.data.train_split < 1:
            raise ValueError(f"train_split must be between 0 and 1, got {self.data.train_split}")
        
        logger.info("Configuration validated successfully")
        logger.info(f"Model: {self.model.n_layer} layers, {self.model.n_head} heads, {self.model.n_embd} embd")
        logger.info(f"Training: {self.training.max_iters} iters, batch={self.training.batch_size}, block={self.model.block_size}")


def load_config(config_path: str = "config.yaml") -> OTTOConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        OTTOConfig instance with loaded values
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.warning(f"Config file {config_path} not found, using defaults")
        return create_default_config()
    
    try:
        with open(config_file, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        model_config = ModelConfig(**yaml_data.get('model', {}))
        training_config = TrainingConfig(**yaml_data.get('training', {}))
        data_config = DataConfig(**yaml_data.get('data', {}))
        paths_config = PathsConfig(**yaml_data.get('paths', {}))
        system_config = SystemConfig(**yaml_data.get('system', {}))
        
        config = OTTOConfig(
            model=model_config,
            training=training_config,
            data=data_config,
            paths=paths_config,
            system=system_config
        )
        
        config.validate()
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        logger.warning("Using default configuration")
        return create_default_config()


def create_default_config() -> OTTOConfig:
    """Create default configuration."""
    config = OTTOConfig(
        model=ModelConfig(),
        training=TrainingConfig(),
        data=DataConfig(),
        paths=PathsConfig(),
        system=SystemConfig()
    )
    config.validate()
    return config


def merge_config_with_args(config: OTTOConfig, **kwargs) -> OTTOConfig:
    """
    Merge configuration with command-line arguments.
    CLI arguments override config file values.
    
    Args:
        config: Base configuration
        **kwargs: CLI arguments to override
        
    Returns:
        Updated configuration
    """
    if 'n_layer' in kwargs and kwargs['n_layer'] is not None:
        config.model.n_layer = kwargs['n_layer']
    if 'n_head' in kwargs and kwargs['n_head'] is not None:
        config.model.n_head = kwargs['n_head']
    if 'n_embd' in kwargs and kwargs['n_embd'] is not None:
        config.model.n_embd = kwargs['n_embd']
    if 'dropout' in kwargs and kwargs['dropout'] is not None:
        config.model.dropout = kwargs['dropout']
    if 'block_size' in kwargs and kwargs['block_size'] is not None:
        config.model.block_size = kwargs['block_size']
    
    if 'max_iters' in kwargs and kwargs['max_iters'] is not None:
        config.training.max_iters = kwargs['max_iters']
    if 'batch_size' in kwargs and kwargs['batch_size'] is not None:
        config.training.batch_size = kwargs['batch_size']
    if 'learning_rate' in kwargs and kwargs['learning_rate'] is not None:
        config.training.learning_rate = kwargs['learning_rate']
    if 'eval_iters' in kwargs and kwargs['eval_iters'] is not None:
        config.training.eval_iters = kwargs['eval_iters']
    
    if 'train_split' in kwargs and kwargs['train_split'] is not None:
        config.data.train_split = kwargs['train_split']
    if 'min_doc_length' in kwargs and kwargs['min_doc_length'] is not None:
        config.data.min_doc_length = kwargs['min_doc_length']
    if 'max_doc_length' in kwargs and kwargs['max_doc_length'] is not None:
        config.data.max_doc_length = kwargs['max_doc_length']
    
    if 'upload_dir' in kwargs and kwargs['upload_dir'] is not None:
        config.paths.upload_dir = kwargs['upload_dir']
    if 'training_data_dir' in kwargs and kwargs['training_data_dir'] is not None:
        config.paths.training_data_dir = kwargs['training_data_dir']
    if 'model_output_dir' in kwargs and kwargs['model_output_dir'] is not None:
        config.paths.model_output_dir = kwargs['model_output_dir']
    
    if 'max_size' in kwargs and kwargs['max_size'] is not None:
        config.system.max_file_size = kwargs['max_size']
    
    config.validate()
    
    logger.info("Merged CLI arguments with config file")
    return config


def save_config(config: OTTOConfig, output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        output_path: Path to save config
    """
    config_dict = {
        'model': asdict(config.model),
        'training': asdict(config.training),
        'data': asdict(config.data),
        'paths': asdict(config.paths),
        'system': asdict(config.system)
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Saved configuration to {output_path}")