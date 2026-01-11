"""
Main training script for LLM fine-tuning.
"""
import os
import sys
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch

# Add src directory to path
sys.path.append(os.path.dirname(__file__))
from config import Config
from data_loader import DataLoader


class LLMFineTuner:
    """Fine-tune a language model on custom data."""
    
    def __init__(self, config: Config):
        """
        Initialize the fine-tuner.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def load_model_and_tokenizer(self):
        """Load the model and tokenizer."""
        print(f"Loading model: {self.config.model.name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_kwargs = {}
        if self.config.model.load_in_8bit:
            model_kwargs['load_in_8bit'] = True
            model_kwargs['device_map'] = 'auto'
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.name,
            **model_kwargs
        )
        
        # Apply LoRA if enabled
        if self.config.lora.enabled:
            try:
                from peft import LoraConfig, get_peft_model
                print("Applying LoRA configuration...")
                
                lora_config = LoraConfig(
                    r=self.config.lora.r,
                    lora_alpha=self.config.lora.lora_alpha,
                    lora_dropout=self.config.lora.lora_dropout,
                    target_modules=self.config.lora.target_modules,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                
                self.model = get_peft_model(self.model, lora_config)
                self.model.print_trainable_parameters()
                
            except ImportError:
                print("Warning: peft library not installed. LoRA will not be applied.")
                print("Install with: pip install peft")
        
        # Ensure model uses the same padding token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        if not self.config.lora.enabled:
            print(f"Model loaded successfully. Parameters: {self.model.num_parameters():,}")

    
    def prepare_data(self):
        """Prepare training and evaluation datasets."""
        print("Preparing datasets...")
        
        data_loader = DataLoader(self.tokenizer, self.config.dataset.max_length)
        
        self.datasets = data_loader.prepare_datasets(
            train_file=self.config.dataset.train_file,
            eval_file=self.config.dataset.eval_file,
            test_file=self.config.dataset.test_file,
            text_column=self.config.dataset.text_column
        )
        
        print(f"Training examples: {len(self.datasets['train'])}")
        if 'validation' in self.datasets:
            print(f"Validation examples: {len(self.datasets['validation'])}")
        if 'test' in self.datasets:
            print(f"Test examples: {len(self.datasets['test'])}")
    
    def setup_training(self):
        """Setup training arguments and trainer."""
        print("Setting up training...")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config.training.output_dir, exist_ok=True)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.config.training.output_dir,
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_steps=self.config.training.warmup_steps,
            logging_steps=self.config.training.logging_steps,
            save_steps=self.config.training.save_steps,
            eval_steps=self.config.training.eval_steps,
            save_total_limit=self.config.training.save_total_limit,
            evaluation_strategy=self.config.training.evaluation_strategy,
            load_best_model_at_end=self.config.training.load_best_model_at_end,
            metric_for_best_model=self.config.training.metric_for_best_model,
            fp16=self.config.training.fp16 and torch.cuda.is_available(),
            report_to="wandb" if self.config.logging.wandb_enabled else "none",
            logging_dir=f"{self.config.training.output_dir}/logs",
        )
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # We're doing causal language modeling, not masked LM
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.datasets['train'],
            eval_dataset=self.datasets.get('validation'),
            data_collator=data_collator,
        )
    
    def train(self):
        """Start the fine-tuning process."""
        print("Starting training...")
        
        # Train the model
        self.trainer.train()
        
        print("Training completed!")
    
    def save_model(self, output_path: str = None):
        """
        Save the fine-tuned model.
        
        Args:
            output_path: Path to save the model. If None, uses config output_dir
        """
        if output_path is None:
            output_path = self.config.training.output_dir
        
        print(f"Saving model to {output_path}")
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print("Model saved successfully!")
    
    def evaluate(self):
        """Evaluate the model on the validation set."""
        if 'validation' not in self.datasets:
            print("No validation dataset available for evaluation.")
            return
        
        print("Evaluating model...")
        eval_results = self.trainer.evaluate()
        
        print("\nEvaluation Results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value:.4f}")
        
        return eval_results
    
    def run_full_pipeline(self):
        """Run the complete fine-tuning pipeline."""
        self.load_model_and_tokenizer()
        self.prepare_data()
        self.setup_training()
        self.train()
        self.save_model()
        
        if 'validation' in self.datasets:
            self.evaluate()


def main():
    """Main entry point for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune an LLM on custom data")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = Config.from_yaml(args.config)
    
    # Initialize and run fine-tuner
    fine_tuner = LLMFineTuner(config)
    fine_tuner.run_full_pipeline()


if __name__ == "__main__":
    main()
