#!/usr/bin/env python3
"""
Validation script to check if the framework is set up correctly.
Run this after installing dependencies to ensure everything works.
"""
import sys
import os


def print_status(message, status="info"):
    """Print colored status messages."""
    colors = {
        "success": "\033[92m✓\033[0m",
        "error": "\033[91m✗\033[0m",
        "warning": "\033[93m!\033[0m",
        "info": "\033[94m→\033[0m"
    }
    print(f"{colors.get(status, '')} {message}")


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n" + "="*60)
    print("Checking Dependencies")
    print("="*60)
    
    required = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("datasets", "HuggingFace Datasets"),
        ("yaml", "PyYAML"),
        ("numpy", "NumPy"),
        ("tqdm", "tqdm")
    ]
    
    all_installed = True
    for module, name in required:
        try:
            __import__(module)
            print_status(f"{name} installed", "success")
        except ImportError:
            print_status(f"{name} NOT installed", "error")
            all_installed = False
    
    return all_installed


def check_files():
    """Check if required files exist."""
    print("\n" + "="*60)
    print("Checking Project Files")
    print("="*60)
    
    required_files = [
        "config.yaml",
        "requirements.txt",
        "src/config.py",
        "src/data_loader.py",
        "src/train.py",
        "src/inference.py",
        "data/train.json",
        "data/eval.json"
    ]
    
    all_exist = True
    for filepath in required_files:
        if os.path.exists(filepath):
            print_status(f"{filepath} exists", "success")
        else:
            print_status(f"{filepath} NOT found", "error")
            all_exist = False
    
    return all_exist


def check_config():
    """Check if configuration is valid."""
    print("\n" + "="*60)
    print("Checking Configuration")
    print("="*60)
    
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print_status("config.yaml is valid YAML", "success")
        print_status(f"Model: {config['model']['name']}", "info")
        print_status(f"Training epochs: {config['training']['num_train_epochs']}", "info")
        print_status(f"Output directory: {config['training']['output_dir']}", "info")
        return True
    except Exception as e:
        print_status(f"Configuration error: {e}", "error")
        return False


def check_data():
    """Check if data files are valid."""
    print("\n" + "="*60)
    print("Checking Data Files")
    print("="*60)
    
    try:
        import json
        
        # Check training data
        with open('data/train.json', 'r') as f:
            train_data = json.load(f)
        print_status(f"Training data: {len(train_data)} examples", "success")
        
        # Check eval data
        with open('data/eval.json', 'r') as f:
            eval_data = json.load(f)
        print_status(f"Evaluation data: {len(eval_data)} examples", "success")
        
        # Check data format
        if train_data and 'text' in train_data[0]:
            print_status("Data format is correct (has 'text' field)", "success")
        else:
            print_status("Data format may be incorrect", "warning")
        
        return True
    except Exception as e:
        print_status(f"Data error: {e}", "error")
        return False


def check_imports():
    """Check if src modules can be imported."""
    print("\n" + "="*60)
    print("Checking Module Imports")
    print("="*60)
    
    sys.path.insert(0, 'src')
    
    modules = [
        ("config", "Config"),
        ("data_loader", "DataLoader"),
        ("train", "LLMFineTuner"),
        ("inference", "LLMInference")
    ]
    
    all_imported = True
    for module_name, class_name in modules:
        try:
            module = __import__(module_name)
            getattr(module, class_name)
            print_status(f"{module_name}.{class_name} imports successfully", "success")
        except Exception as e:
            print_status(f"{module_name}.{class_name} import error: {e}", "error")
            all_imported = False
    
    return all_imported


def check_gpu():
    """Check GPU availability."""
    print("\n" + "="*60)
    print("Checking GPU Support")
    print("="*60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print_status(f"CUDA available: {torch.cuda.get_device_name(0)}", "success")
            print_status(f"CUDA version: {torch.version.cuda}", "info")
            return True
        else:
            print_status("CUDA not available (will use CPU)", "warning")
            return False
    except:
        print_status("Cannot check GPU status", "warning")
        return False


def main():
    """Run all validation checks."""
    print("\n" + "="*60)
    print("LLM Fine-Tuning Framework Validation")
    print("="*60)
    
    results = {
        "Files": check_files(),
        "Configuration": check_config(),
        "Data": check_data(),
        "Dependencies": check_dependencies(),
        "GPU": check_gpu(),
        "Imports": check_imports()
    }
    
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)
    
    all_passed = True
    for check, passed in results.items():
        if passed:
            print_status(f"{check}: PASSED", "success")
        elif check == "GPU":  # GPU is optional
            print_status(f"{check}: Optional (will use CPU)", "warning")
        else:
            print_status(f"{check}: FAILED", "error")
            all_passed = False
    
    print("\n" + "="*60)
    
    if all_passed:
        print_status("All critical checks passed! ✨", "success")
        print_status("You're ready to fine-tune models!", "success")
        print("\nNext steps:")
        print("  1. Review config.yaml")
        print("  2. Prepare your data in data/")
        print("  3. Run: python run_finetuning.py")
        return 0
    else:
        print_status("Some checks failed. Please fix the issues above.", "error")
        print("\nTo install dependencies, run:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
