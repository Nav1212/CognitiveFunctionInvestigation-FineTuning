"""
Example script showing how to use the LLM fine-tuning framework.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import Config, ModelConfig, TrainingConfig, DatasetConfig
from src.train import LLMFineTuner
from src.inference import LLMInference


def example_1_basic_finetuning():
    """Example 1: Basic fine-tuning with config file."""
    print("Example 1: Basic Fine-tuning")
    print("-" * 50)
    
    # Load configuration from YAML file
    config = Config.from_yaml("config.yaml")
    
    # Run fine-tuning
    fine_tuner = LLMFineTuner(config)
    fine_tuner.run_full_pipeline()


def example_2_custom_config():
    """Example 2: Fine-tuning with custom configuration."""
    print("\nExample 2: Custom Configuration")
    print("-" * 50)
    
    # Create custom configuration programmatically
    config = Config()
    config.model.name = "gpt2"
    config.training.num_train_epochs = 2
    config.training.per_device_train_batch_size = 2
    config.training.output_dir = "./models/custom_model"
    config.dataset.train_file = "data/train.json"
    config.dataset.eval_file = "data/eval.json"
    
    # Run fine-tuning
    fine_tuner = LLMFineTuner(config)
    fine_tuner.run_full_pipeline()


def example_3_inference():
    """Example 3: Running inference with a fine-tuned model."""
    print("\nExample 3: Inference")
    print("-" * 50)
    
    model_path = "./models/fine_tuned_model"
    
    # Initialize inference engine
    inference = LLMInference(model_path)
    
    # Generate text from prompts
    prompts = [
        "Question: What is cognitive function?",
        "Question: How does exercise affect the brain?"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        responses = inference.generate(prompt, max_length=150, num_return_sequences=1)
        print(f"Response: {responses[0]}\n")


def example_4_step_by_step():
    """Example 4: Step-by-step fine-tuning process."""
    print("\nExample 4: Step-by-Step Process")
    print("-" * 50)
    
    # Load config
    config = Config.from_yaml("config.yaml")
    fine_tuner = LLMFineTuner(config)
    
    # Step 1: Load model and tokenizer
    print("\nStep 1: Loading model and tokenizer...")
    fine_tuner.load_model_and_tokenizer()
    
    # Step 2: Prepare data
    print("\nStep 2: Preparing datasets...")
    fine_tuner.prepare_data()
    
    # Step 3: Setup training
    print("\nStep 3: Setting up training...")
    fine_tuner.setup_training()
    
    # Step 4: Train
    print("\nStep 4: Training...")
    fine_tuner.train()
    
    # Step 5: Save model
    print("\nStep 5: Saving model...")
    fine_tuner.save_model()
    
    # Step 6: Evaluate
    print("\nStep 6: Evaluating...")
    fine_tuner.evaluate()


if __name__ == "__main__":
    # Choose which example to run
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LLM fine-tuning examples")
    parser.add_argument(
        "--example",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Which example to run (1-4)"
    )
    
    args = parser.parse_args()
    
    examples = {
        1: example_1_basic_finetuning,
        2: example_2_custom_config,
        3: example_3_inference,
        4: example_4_step_by_step
    }
    
    examples[args.example]()
