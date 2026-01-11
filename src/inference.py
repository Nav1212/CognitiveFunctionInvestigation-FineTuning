"""
Inference script for using a fine-tuned model.
"""
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional

# Add src directory to path
sys.path.append(os.path.dirname(__file__))


class LLMInference:
    """Run inference with a fine-tuned language model."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the fine-tuned model directory
            device: Device to run inference on ('cpu', 'cuda', or 'auto')
        """
        self.model_path = model_path
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading model from {model_path}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 200,
        num_return_sequences: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            num_return_sequences: Number of sequences to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            **kwargs: Additional arguments for generation
            
        Returns:
            List of generated text strings
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode outputs
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return generated_texts
    
    def interactive_mode(self):
        """Run an interactive chat session with the model."""
        print("\n" + "="*50)
        print("Interactive Mode - Type 'quit' or 'exit' to stop")
        print("="*50 + "\n")
        
        while True:
            try:
                prompt = input("You: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not prompt:
                    continue
                
                # Generate response
                responses = self.generate(prompt, num_return_sequences=1)
                response = responses[0]
                
                # Print only the generated part (after the prompt)
                if response.startswith(prompt):
                    response = response[len(prompt):].strip()
                
                print(f"\nModel: {response}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main entry point for inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/fine_tuned_model",
        help="Path to the fine-tuned model directory"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for generation (if not provided, enters interactive mode)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=200,
        help="Maximum length of generated text"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=1,
        help="Number of sequences to generate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run inference on"
    )
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = LLMInference(args.model_path, device=args.device)
    
    if args.prompt:
        # Single generation
        print(f"\nPrompt: {args.prompt}\n")
        responses = inference.generate(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            num_return_sequences=args.num_sequences
        )
        
        print("Generated responses:")
        for i, response in enumerate(responses, 1):
            print(f"\n[Response {i}]")
            print(response)
            print("-" * 50)
    else:
        # Interactive mode
        inference.interactive_mode()


if __name__ == "__main__":
    main()
