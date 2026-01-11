"""
Configuration management for LLM fine-tuning.
"""
import yaml
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for model settings."""
    name: str = "gpt2"
    load_in_8bit: bool = False
    device: str = "auto"


@dataclass
class TrainingConfig:
    """Configuration for training settings."""
    output_dir: str = "./models/fine_tuned_model"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2
    fp16: bool = False
    evaluation_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"


@dataclass
class DatasetConfig:
    """Configuration for dataset settings."""
    train_file: str = "data/train.json"
    eval_file: str = "data/eval.json"
    test_file: str = "data/test.json"
    text_column: str = "text"
    max_length: int = 512


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) settings."""
    enabled: bool = False
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class LoggingConfig:
    """Configuration for logging settings."""
    wandb_enabled: bool = False
    wandb_project: str = "llm-finetuning"


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Config object
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            dataset=DatasetConfig(**config_dict.get('dataset', {})),
            lora=LoRAConfig(**config_dict.get('lora', {})),
            logging=LoggingConfig(**config_dict.get('logging', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'dataset': self.dataset.__dict__,
            'lora': self.lora.__dict__,
            'logging': self.logging.__dict__
        }
