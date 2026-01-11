# Quick Reference Guide

## Installation
```bash
pip install -r requirements.txt
```

## Quick Commands

### Training
```bash
# Default training
python run_finetuning.py

# With custom config
python src/train.py --config my_config.yaml
```

### Inference
```bash
# Interactive mode
python src/inference.py --model_path ./models/fine_tuned_model

# Single prompt
python src/inference.py --model_path ./models/fine_tuned_model --prompt "Your question here"
```

### Testing
```bash
python -m pytest tests/
```

## Configuration Quick Edit

Edit `config.yaml`:

```yaml
# Change model
model:
  name: "gpt2"  # Options: gpt2, distilgpt2, microsoft/phi-2, etc.

# Adjust training
training:
  num_train_epochs: 3       # Number of epochs
  learning_rate: 2.0e-5     # Learning rate
  per_device_train_batch_size: 4  # Batch size

# Update data paths
dataset:
  train_file: "data/train.json"
  eval_file: "data/eval.json"
```

## Data Format

JSON file with text entries:
```json
[
  {"text": "Question: Your question?\nAnswer: Your answer."},
  {"text": "Another example text..."}
]
```

## Common Issues

**Out of Memory?**
- Reduce batch size in config.yaml
- Enable `load_in_8bit: true` in model config
- Reduce `max_length` in dataset config

**Training too slow?**
- Enable `fp16: true` if using GPU
- Increase batch size if you have memory
- Use smaller model (e.g., distilgpt2)

**Model not learning?**
- Increase training epochs
- Add more training data
- Adjust learning rate

## Directory Structure
```
├── config.yaml          # Configuration
├── data/               # Your datasets
├── src/                # Source code
├── models/             # Saved models (after training)
└── examples/           # Example scripts
```
