"""
Data loading and preprocessing utilities for LLM fine-tuning.
"""
import json
from typing import Dict, List, Optional
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer


class DataLoader:
    """Load and preprocess datasets for LLM fine-tuning."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        """
        Initialize the data loader.
        
        Args:
            tokenizer: The tokenizer to use for preprocessing
            max_length: Maximum sequence length for tokenization
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = 'text'  # Will be set by prepare_datasets
    
    def load_json_data(self, file_path: str) -> List[Dict]:
        """
        Load data from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of data examples
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def preprocess_function(self, examples: Dict) -> Dict:
        """
        Preprocess examples for training.
        
        Args:
            examples: Batch of examples with text field
            
        Returns:
            Tokenized examples
        """
        # Tokenize the texts using the configured column name
        tokenized = self.tokenizer(
            examples[self.text_column],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors=None
        )
        
        # For causal language modeling, labels are the same as input_ids
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
    
    def prepare_datasets(
        self,
        train_file: str,
        eval_file: Optional[str] = None,
        test_file: Optional[str] = None,
        text_column: str = 'text'
    ) -> DatasetDict:
        """
        Prepare datasets for training.
        
        Args:
            train_file: Path to training data file
            eval_file: Path to evaluation data file (optional)
            test_file: Path to test data file (optional)
            text_column: Name of the column containing text data
            
        Returns:
            DatasetDict containing train, eval, and optionally test splits
        """
        # Store the text column name for use in preprocessing
        self.text_column = text_column
        
        # Load training data
        train_data = self.load_json_data(train_file)
        train_dataset = Dataset.from_list(train_data)
        
        datasets = {'train': train_dataset}
        
        # Load evaluation data if provided
        if eval_file:
            eval_data = self.load_json_data(eval_file)
            eval_dataset = Dataset.from_list(eval_data)
            datasets['validation'] = eval_dataset
        
        # Load test data if provided
        if test_file:
            test_data = self.load_json_data(test_file)
            test_dataset = Dataset.from_list(test_data)
            datasets['test'] = test_dataset
        
        # Convert to DatasetDict
        dataset_dict = DatasetDict(datasets)
        
        # Apply preprocessing
        tokenized_datasets = dataset_dict.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset_dict['train'].column_names,
            desc="Tokenizing datasets"
        )
        
        return tokenized_datasets


def load_custom_dataset(file_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512) -> Dataset:
    """
    Convenience function to load a single dataset file.
    
    Args:
        file_path: Path to the JSON file
        tokenizer: Tokenizer for preprocessing
        max_length: Maximum sequence length
        
    Returns:
        Preprocessed Dataset
    """
    loader = DataLoader(tokenizer, max_length)
    data = loader.load_json_data(file_path)
    dataset = Dataset.from_list(data)
    
    tokenized_dataset = dataset.map(
        loader.preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset
