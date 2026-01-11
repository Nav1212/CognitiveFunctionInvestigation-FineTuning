# LLM Fine-Tuning Framework for Cognitive Function Investigation

A comprehensive framework for fine-tuning Large Language Models (LLMs) using custom datasets. This project investigates how fine-tuning can affect AI behavior and performance, with a focus on cognitive function-related applications.

## Features

- 🚀 **Easy-to-use**: Simple configuration-based setup
- 📊 **Flexible Data Loading**: Support for JSON datasets with custom formats
- 🎯 **Multiple Model Support**: Works with any HuggingFace transformer model
- ⚙️ **Configurable Training**: YAML-based configuration for all training parameters
- 💾 **Model Management**: Automatic model saving and loading
- 🔮 **Interactive Inference**: Built-in inference engine with interactive mode
- 📈 **Monitoring**: Support for logging and evaluation metrics
- 🧪 **Well-tested**: Includes unit tests and usage examples

## Project Structure

```
CognitiveFunctionInvestigation-FineTuning/
├── config.yaml              # Main configuration file
├── requirements.txt         # Python dependencies
├── run_finetuning.py       # Quick start script
├── data/                    # Dataset directory
│   ├── train.json          # Training data
│   ├── eval.json           # Evaluation data
│   └── test.json           # Test data
├── src/                     # Source code
│   ├── config.py           # Configuration management
│   ├── data_loader.py      # Data loading utilities
│   ├── train.py            # Training script
│   └── inference.py        # Inference script
├── examples/                # Usage examples
│   └── usage_examples.py   # Example scripts
├── tests/                   # Unit tests
│   └── test_framework.py   # Framework tests
└── models/                  # Saved models (created during training)
```

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Nav1212/CognitiveFunctionInvestigation-FineTuning.git
cd CognitiveFunctionInvestigation-FineTuning
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Fine-Tuning

The easiest way to get started is using the provided script:

```bash
python run_finetuning.py
```

This will:
- Load the configuration from `config.yaml`
- Load the GPT-2 model
- Fine-tune on the provided cognitive function dataset
- Save the model to `./models/fine_tuned_model`

### 2. Custom Configuration

Edit `config.yaml` to customize your training:

```yaml
model:
  name: "gpt2"  # Change to any HuggingFace model
  
training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  learning_rate: 2.0e-5
  output_dir: "./models/fine_tuned_model"

dataset:
  train_file: "data/train.json"
  eval_file: "data/eval.json"
  max_length: 512
```

### 3. Using Custom Data

Your data should be in JSON format with a `text` field:

```json
[
  {
    "text": "Question: What is cognitive function?\nAnswer: Cognitive function refers to mental processes..."
  },
  {
    "text": "Question: How does memory work?\nAnswer: Memory is the process by which information..."
  }
]
```

Place your data files in the `data/` directory and update paths in `config.yaml`.

## Usage

### Training from Command Line

```bash
# Use default config
python src/train.py

# Use custom config
python src/train.py --config my_config.yaml
```

### Running Inference

```bash
# Interactive mode
python src/inference.py --model_path ./models/fine_tuned_model

# Single prompt
python src/inference.py \
  --model_path ./models/fine_tuned_model \
  --prompt "Question: What affects cognitive decline?" \
  --max_length 200 \
  --temperature 0.7
```

### Programmatic Usage

```python
from src.config import Config
from src.train import LLMFineTuner
from src.inference import LLMInference

# Training
config = Config.from_yaml("config.yaml")
fine_tuner = LLMFineTuner(config)
fine_tuner.run_full_pipeline()

# Inference
inference = LLMInference("./models/fine_tuned_model")
response = inference.generate("Question: What is neuroplasticity?")
print(response[0])
```

## Advanced Features

### Parameter-Efficient Fine-Tuning (LoRA)

Enable LoRA in `config.yaml` for memory-efficient training:

```yaml
lora:
  enabled: true
  r: 8
  lora_alpha: 32
  lora_dropout: 0.1
```

### Experiment Tracking

Enable Weights & Biases tracking:

```yaml
logging:
  wandb_enabled: true
  wandb_project: "cognitive-function-llm"
```

### Custom Training Arguments

Modify training parameters in `config.yaml`:

```yaml
training:
  gradient_accumulation_steps: 4  # Effective batch size multiplier
  fp16: true                       # Use mixed precision (requires CUDA)
  warmup_steps: 500               # Learning rate warmup
  save_steps: 1000                # Save checkpoint every N steps
```

## Examples

Check out the `examples/` directory for detailed examples:

```bash
# Run example 1: Basic fine-tuning
python examples/usage_examples.py --example 1

# Run example 2: Custom configuration
python examples/usage_examples.py --example 2

# Run example 3: Inference
python examples/usage_examples.py --example 3

# Run example 4: Step-by-step process
python examples/usage_examples.py --example 4
```

## Testing

Run the test suite:

```bash
python -m pytest tests/
# or
python tests/test_framework.py
```

## Dataset Information

The provided dataset (`data/`) contains sample questions and answers about cognitive function, including topics like:
- Cognitive decline and aging
- Alzheimer's disease
- Memory and learning
- Executive function
- Neuroplasticity
- Stress and cognitive performance

You can replace this with your own domain-specific data.

## Model Support

This framework supports any causal language model from HuggingFace, including:

- **GPT-2**: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
- **GPT-Neo/GPT-J**: `EleutherAI/gpt-neo-125M`, `EleutherAI/gpt-j-6B`
- **OPT**: `facebook/opt-125m`, `facebook/opt-350m`
- **BLOOM**: `bigscience/bloom-560m`, `bigscience/bloom-1b1`
- **LLaMA**: `meta-llama/Llama-2-7b-hf` (requires access)
- **Phi**: `microsoft/phi-2`
- And many more!

## Performance Tips

1. **Memory optimization**: Enable 8-bit loading for large models
2. **Batch size**: Adjust based on your GPU memory
3. **Gradient accumulation**: Use to simulate larger batch sizes
4. **Mixed precision**: Enable fp16 for faster training on CUDA
5. **Dataset size**: Ensure sufficient training data for your domain

## Troubleshooting

### Out of Memory Errors
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Enable `load_in_8bit` in model config
- Reduce `max_length` in dataset config

### Slow Training
- Enable `fp16` if using CUDA
- Reduce `logging_steps` and `save_steps`
- Use a smaller model variant

### Poor Model Performance
- Increase training epochs
- Adjust learning rate
- Add more training data
- Try different model architectures

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{cognitive_function_finetuning,
  title={LLM Fine-Tuning Framework for Cognitive Function Investigation},
  author={Nav1212},
  year={2026},
  url={https://github.com/Nav1212/CognitiveFunctionInvestigation-FineTuning}
}
```

## Acknowledgments

- Built with [HuggingFace Transformers](https://huggingface.co/transformers/)
- Inspired by research in cognitive function and AI applications in healthcare 
