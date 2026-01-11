"""
Unit tests for the LLM fine-tuning framework.
"""
import unittest
import os
import sys
import json
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import Config, ModelConfig, TrainingConfig, DatasetConfig
from data_loader import DataLoader
from transformers import AutoTokenizer


class TestConfig(unittest.TestCase):
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = Config()
        self.assertEqual(config.model.name, "gpt2")
        self.assertEqual(config.training.num_train_epochs, 3)
        self.assertEqual(config.dataset.max_length, 512)
    
    def test_config_from_dict(self):
        """Test config creation from dict."""
        model_config = ModelConfig(name="distilgpt2")
        self.assertEqual(model_config.name, "distilgpt2")
    
    def test_config_to_dict(self):
        """Test config conversion to dict."""
        config = Config()
        config_dict = config.to_dict()
        self.assertIn('model', config_dict)
        self.assertIn('training', config_dict)
        self.assertIn('dataset', config_dict)


class TestDataLoader(unittest.TestCase):
    """Test data loading and preprocessing."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample data file
        self.sample_data = [
            {"text": "This is a test sentence."},
            {"text": "Another test sentence for training."}
        ]
        
        self.data_file = os.path.join(self.temp_dir, "test_data.json")
        with open(self.data_file, 'w') as f:
            json.dump(self.sample_data, f)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_load_json_data(self):
        """Test JSON data loading."""
        loader = DataLoader(self.tokenizer)
        data = loader.load_json_data(self.data_file)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]['text'], "This is a test sentence.")
    
    def test_preprocess_function(self):
        """Test preprocessing function."""
        loader = DataLoader(self.tokenizer, max_length=128)
        
        examples = {'text': [item['text'] for item in self.sample_data]}
        processed = loader.preprocess_function(examples)
        
        self.assertIn('input_ids', processed)
        self.assertIn('attention_mask', processed)
        self.assertIn('labels', processed)
    
    def test_prepare_datasets(self):
        """Test dataset preparation."""
        loader = DataLoader(self.tokenizer, max_length=128)
        
        datasets = loader.prepare_datasets(
            train_file=self.data_file,
            eval_file=self.data_file
        )
        
        self.assertIn('train', datasets)
        self.assertIn('validation', datasets)
        self.assertEqual(len(datasets['train']), 2)


class TestIntegration(unittest.TestCase):
    """Integration tests for the framework."""
    
    def test_config_yaml_loading(self):
        """Test loading configuration from YAML file."""
        # This assumes config.yaml exists in the project root
        config_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'config.yaml'
        )
        
        if os.path.exists(config_path):
            config = Config.from_yaml(config_path)
            self.assertIsInstance(config.model, ModelConfig)
            self.assertIsInstance(config.training, TrainingConfig)
            self.assertIsInstance(config.dataset, DatasetConfig)


if __name__ == '__main__':
    unittest.main()
