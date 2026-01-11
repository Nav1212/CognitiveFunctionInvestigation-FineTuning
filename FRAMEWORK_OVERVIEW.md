# Framework Overview

## What This Framework Does

This is a complete, production-ready framework for fine-tuning Large Language Models (LLMs) on custom datasets. It provides a simple, configuration-driven approach to:

1. **Load and preprocess** custom text datasets
2. **Fine-tune** any HuggingFace transformer model
3. **Evaluate** model performance
4. **Run inference** on the fine-tuned model

## Key Components

### 1. Configuration System (`src/config.py`)
- **Purpose**: Centralized configuration management
- **Features**: 
  - YAML-based configuration
  - Dataclass-based structure
  - Support for model, training, dataset, and LoRA settings
- **Usage**: `config = Config.from_yaml('config.yaml')`

### 2. Data Loader (`src/data_loader.py`)
- **Purpose**: Load and preprocess training data
- **Features**:
  - JSON data file support
  - Automatic tokenization
  - Support for train/eval/test splits
  - Configurable sequence length
- **Usage**: Automatically used by the training pipeline

### 3. Training Module (`src/train.py`)
- **Purpose**: Fine-tune language models
- **Features**:
  - Full training pipeline
  - Automatic model saving
  - Evaluation support
  - Progress logging
  - Support for LoRA (parameter-efficient fine-tuning)
- **Usage**: `python src/train.py --config config.yaml`

### 4. Inference Module (`src/inference.py`)
- **Purpose**: Generate text with fine-tuned models
- **Features**:
  - Interactive chat mode
  - Batch generation
  - Configurable sampling parameters
  - GPU/CPU support
- **Usage**: `python src/inference.py --model_path ./models/fine_tuned_model`

## Architecture

```
User Input (Prompt/Question)
         ↓
[Configuration] ← config.yaml
         ↓
[Data Loader] ← data/*.json
         ↓
[Tokenizer] ← HuggingFace
         ↓
[Model Loading] ← Pretrained LLM
         ↓
[Fine-Tuning] ← Training Arguments
         ↓
[Trained Model] → ./models/
         ↓
[Inference] → Generated Text
```

## Data Flow

### Training Flow
1. Load configuration from YAML
2. Initialize model and tokenizer
3. Load and tokenize dataset
4. Create training arguments
5. Initialize Trainer
6. Run training loop
7. Save fine-tuned model
8. Evaluate on validation set

### Inference Flow
1. Load fine-tuned model
2. Tokenize input prompt
3. Generate output with sampling
4. Decode and return text

## Configuration Options

### Model Settings
- `name`: HuggingFace model identifier
- `load_in_8bit`: Memory-efficient loading
- `device`: CPU/GPU selection

### Training Settings
- `num_train_epochs`: Number of training epochs
- `batch_size`: Samples per batch
- `learning_rate`: Optimizer learning rate
- `gradient_accumulation`: Effective batch size multiplier
- `fp16`: Mixed precision training

### Dataset Settings
- `train_file`, `eval_file`, `test_file`: Data paths
- `text_column`: Column name for text data
- `max_length`: Maximum sequence length

### LoRA Settings (Optional)
- `enabled`: Use parameter-efficient fine-tuning
- `r`: LoRA rank
- `lora_alpha`: LoRA scaling parameter

## Supported Models

Any causal language model from HuggingFace:
- GPT-2 family (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
- GPT-Neo/GPT-J (EleutherAI models)
- OPT (Facebook models)
- BLOOM (BigScience models)
- LLaMA (Meta models - requires access)
- Phi (Microsoft models)
- And hundreds more...

## Data Format

The framework expects JSON files with text entries:

```json
[
  {
    "text": "Your training text here. Can be Q&A pairs, articles, etc."
  },
  {
    "text": "Another training example..."
  }
]
```

**Tips:**
- Include diverse examples
- Format consistently (e.g., "Question: ... Answer: ...")
- Aim for 100+ examples minimum
- Include validation data for monitoring

## Extending the Framework

### Adding Custom Data Processing
Modify `src/data_loader.py`:
```python
def custom_preprocess(self, examples):
    # Your custom preprocessing logic
    return processed_examples
```

### Changing Model Architecture
Update `src/train.py` to use different model types:
```python
from transformers import AutoModelForSeq2SeqLM  # For T5, BART, etc.
```

### Custom Training Logic
Override trainer methods in `src/train.py`:
```python
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Custom loss computation
        pass
```

## Performance Optimization

### Memory Optimization
1. Use smaller models (distilgpt2 instead of gpt2-xl)
2. Enable 8-bit quantization (`load_in_8bit: true`)
3. Reduce batch size
4. Reduce max sequence length
5. Use gradient accumulation

### Speed Optimization
1. Enable mixed precision (`fp16: true`)
2. Use larger batch sizes (if memory allows)
3. Reduce logging frequency
4. Use faster models (distilled versions)
5. Use GPU when available

### Quality Optimization
1. More training data
2. More training epochs
3. Better data quality and diversity
4. Appropriate learning rate
5. Validation-based early stopping

## Best Practices

1. **Always start with a small model** (gpt2, distilgpt2) for testing
2. **Use a validation set** to monitor training progress
3. **Save checkpoints regularly** via `save_steps`
4. **Monitor eval_loss** to detect overfitting
5. **Test inference early** to verify model behavior
6. **Version your datasets** for reproducibility
7. **Document your configuration** changes
8. **Start with default hyperparameters** then tune

## Troubleshooting Guide

### Training Issues
- **Loss not decreasing**: Increase learning rate or check data quality
- **Loss exploding**: Decrease learning rate
- **Slow convergence**: Increase batch size or learning rate
- **Overfitting**: Add more data, reduce epochs, or use regularization

### Technical Issues
- **CUDA out of memory**: Reduce batch size or enable 8-bit loading
- **Slow training**: Enable fp16, increase batch size
- **Import errors**: Install dependencies: `pip install -r requirements.txt`
- **Model not loading**: Check model name and internet connection

## Use Cases

This framework is ideal for:

1. **Domain-specific chatbots** (medical, legal, technical support)
2. **Question-answering systems** (FAQ automation)
3. **Content generation** (marketing, creative writing)
4. **Code generation** (documentation, boilerplate)
5. **Text classification** (with prompt engineering)
6. **Data augmentation** (generating synthetic examples)
7. **Research experiments** (cognitive function, clinical applications)

## Next Steps

After setting up the framework:

1. **Collect your data** in the required JSON format
2. **Update config.yaml** with your preferences
3. **Run training**: `python run_finetuning.py`
4. **Test your model**: `python src/inference.py`
5. **Iterate**: Adjust config, add data, retrain
6. **Deploy**: Integrate inference into your application

## Resources

- HuggingFace Transformers: https://huggingface.co/docs/transformers
- Model Hub: https://huggingface.co/models
- Dataset Format Examples: See `data/` directory
- Configuration Examples: See `config.yaml`
- Code Examples: See `examples/` directory

## Support

For issues or questions:
1. Check the README.md
2. Review QUICKSTART.md
3. Run the examples in `examples/`
4. Check the tests in `tests/`
5. Open an issue on GitHub
