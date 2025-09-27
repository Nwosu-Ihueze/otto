import torch
import torch.nn.functional as F
import tiktoken
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time

from otto.models.gpt import GPT, GPTConfig  # Your model classes

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference parameters."""
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: float = 1.0
    do_sample: bool = True


class SLMInference:
    """
    Inference engine for trained Small Language Models.
    
    Handles model loading, text generation, and evaluation.
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = None
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to saved model state dict
            config_path: Path to model config (if separate from checkpoint)
            device: Device to run inference on
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        
        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.tokenizer.max_token_value + 1
        
        # Load model
        self.model, self.config = self._load_model(config_path)
        self.model.eval()
        
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Model config: {self.config}")
        logger.info(f"Device: {self.device}")
    
    def _load_model(self, config_path: Optional[str] = None) -> Tuple[GPT, GPTConfig]:
        """Load model and configuration."""
        try:
       
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
      
                config = checkpoint['model_config']
                model = GPT(config)
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Loaded full checkpoint with config")
                logger.info(f"Model config: {config}")
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        
                state_dict = checkpoint['model_state_dict']
                config = self._infer_config_from_state_dict(state_dict)
                model = GPT(config)
                model.load_state_dict(state_dict)
                logger.info("Inferred config from checkpoint state dict")
                logger.info(f"Inferred config: {config}")
            else:
           
                config = self._infer_config_from_state_dict(checkpoint)
                model = GPT(config)
                model.load_state_dict(checkpoint)
                logger.info("Inferred config from state dict")
                logger.info(f"Inferred config: {config}")
            
            model = model.to(self.device)
            return model, config
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")
    
    def _infer_config_from_state_dict(self, state_dict: dict) -> GPTConfig:
        """Infer model configuration from state dict."""
   
        n_embd = state_dict['transformer.wte.weight'].shape[1]
        
    
        vocab_size = state_dict['transformer.wte.weight'].shape[0]
        

        block_size = state_dict['transformer.wpe.weight'].shape[0]
        
 
        n_layer = 0
        for key in state_dict.keys():
            if key.startswith('transformer.h.') and key.endswith('.ln1.weight'):
                layer_num = int(key.split('.')[2])
                n_layer = max(n_layer, layer_num + 1)
        
        # Get number of heads from attention weights
        # c_attn weight shape is [3 * n_embd, n_embd], so 3 * n_embd / n_embd = 3
        # But we need to infer n_head from the fact that n_embd must be divisible by n_head
        # Look at a transformer block's attention output projection
        attn_weight = state_dict['transformer.h.0.attn.c_attn.weight']
        total_dim = attn_weight.shape[0]  # This is 3 * n_embd
        
     
        assert total_dim == 3 * n_embd, f"Expected {3 * n_embd}, got {total_dim}"
        
      
        possible_heads = [1, 2, 4, 6, 8, 12, 16]
        n_head = None
        for h in possible_heads:
            if n_embd % h == 0:
                n_head = h
        
 
        if n_head is None:
            n_head = min(4, n_embd // 64)  
        
  
        bias = 'transformer.h.0.ln1.bias' in state_dict
        
        config = GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=0.0,  
            bias=bias
        )
        
        logger.info(f"Inferred config: vocab_size={vocab_size}, block_size={block_size}, "
                   f"n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}, bias={bias}")
        
        return config
    
    def generate_text(
        self,
        prompt: str = "",
        inference_config: InferenceConfig = None
    ) -> Dict[str, any]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            inference_config: Generation parameters
            
        Returns:
            Dictionary with generated text and metadata
        """
        if inference_config is None:
            inference_config = InferenceConfig()
        
        start_time = time.time()
        
  
        if prompt:
            prompt_tokens = self.tokenizer.encode_ordinary(prompt)
            context = torch.tensor(prompt_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        else:
    
            context = torch.randint(0, self.vocab_size, (1, 1), device=self.device)
        
 
        with torch.no_grad():
            generated = self.model.generate(
                context,
                max_new_tokens=inference_config.max_new_tokens,
                temperature=inference_config.temperature,
                top_k=inference_config.top_k
            )
        
    
        generated_tokens = generated[0].tolist()
        full_text = self.tokenizer.decode(generated_tokens)
        

        if prompt:
            new_tokens = generated_tokens[len(prompt_tokens):]
            generated_text = self.tokenizer.decode(new_tokens)
        else:
            generated_text = full_text
        
        generation_time = time.time() - start_time
        tokens_per_second = inference_config.max_new_tokens / generation_time
        
        return {
            'prompt': prompt,
            'generated_text': generated_text,
            'full_text': full_text,
            'prompt_tokens': len(prompt_tokens) if prompt else 0,
            'generated_tokens': len(new_tokens) if prompt else len(generated_tokens),
            'total_tokens': len(generated_tokens),
            'generation_time': generation_time,
            'tokens_per_second': tokens_per_second,
            'config': inference_config
        }
    
    def batch_generate(
        self,
        prompts: List[str],
        inference_config: InferenceConfig = None
    ) -> List[Dict[str, any]]:
        """Generate text for multiple prompts."""
        results = []
        for prompt in prompts:
            result = self.generate_text(prompt, inference_config)
            results.append(result)
        return results
    
    def interactive_generation(self):
        """Interactive text generation session."""
        print("=== SLM Interactive Generation ===")
        print("Type prompts to generate text. Type 'quit' to exit.")
        print("Commands:")
        print("  - 'quit': Exit")
        print("  - 'config': Show current config")
        print("  - 'temp X': Set temperature to X")
        print("  - 'tokens X': Set max tokens to X")
        print("-" * 50)
        
        config = InferenceConfig()
        
        while True:
            try:
                user_input = input("\nPrompt: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'config':
                    print(f"Current config: {config}")
                    continue
                elif user_input.lower().startswith('temp '):
                    try:
                        temp = float(user_input.split()[1])
                        config.temperature = temp
                        print(f"Temperature set to {temp}")
                    except:
                        print("Invalid temperature value")
                    continue
                elif user_input.lower().startswith('tokens '):
                    try:
                        tokens = int(user_input.split()[1])
                        config.max_new_tokens = tokens
                        print(f"Max tokens set to {tokens}")
                    except:
                        print("Invalid token count")
                    continue
                
   
                result = self.generate_text(user_input, config)
                
                print(f"\nGenerated ({result['generation_time']:.2f}s, {result['tokens_per_second']:.1f} tok/s):")
                print(result['generated_text'])
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


class SLMEvaluator:
    """
    Evaluation utilities for Small Language Models.
    """
    
    def __init__(self, inference_engine: SLMInference):
        """
        Initialize evaluator.
        
        Args:
            inference_engine: SLMInference instance
        """
        self.inference = inference_engine
    
    def perplexity_evaluation(
        self,
        test_data_path: str,
        max_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Calculate perplexity on test data.
        
        Args:
            test_data_path: Path to test data (binary file)
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Dictionary with perplexity metrics
        """
        logger.info(f"Calculating perplexity on {test_data_path}")
        
      
        data = np.memmap(test_data_path, dtype=np.uint16, mode='r')
        

        block_size = self.inference.config.block_size
        num_samples = min(max_samples, len(data) // block_size)
        
        total_loss = 0.0
        total_tokens = 0
        
        self.inference.model.eval()
        with torch.no_grad():
            for i in range(num_samples):
   
                start_idx = np.random.randint(0, len(data) - block_size)
                sequence = torch.from_numpy(
                    data[start_idx:start_idx + block_size].astype(np.int64)
                ).unsqueeze(0).to(self.inference.device)
                

                logits, loss = self.inference.model(sequence, sequence)
                total_loss += loss.item()
                total_tokens += block_size
        
        avg_loss = total_loss / num_samples
        perplexity = np.exp(avg_loss)
        
        return {
            'perplexity': perplexity,
            'avg_loss': avg_loss,
            'num_samples': num_samples,
            'total_tokens': total_tokens
        }
    
    def prompt_evaluation(
        self,
        test_prompts: List[str],
        inference_config: InferenceConfig = None
    ) -> Dict[str, any]:
        """
        Evaluate model on a set of test prompts.
        
        Args:
            test_prompts: List of test prompts
            inference_config: Generation configuration
            
        Returns:
            Evaluation results
        """
        results = self.inference.batch_generate(test_prompts, inference_config)
        

        generation_times = [r['generation_time'] for r in results]
        tokens_per_second = [r['tokens_per_second'] for r in results]
        generated_lengths = [r['generated_tokens'] for r in results]
        
        return {
            'results': results,
            'avg_generation_time': np.mean(generation_times),
            'avg_tokens_per_second': np.mean(tokens_per_second),
            'avg_generated_length': np.mean(generated_lengths),
            'total_prompts': len(test_prompts)
        }
    
    def coherence_evaluation(
        self,
        prompts: List[str],
        inference_config: InferenceConfig = None
    ) -> Dict[str, any]:
        """
        Basic coherence evaluation using repetition and diversity metrics.
        
        Args:
            prompts: Test prompts
            inference_config: Generation configuration
            
        Returns:
            Coherence metrics
        """
        results = self.inference.batch_generate(prompts, inference_config)
        
        repetition_scores = []
        diversity_scores = []
        
        for result in results:
            text = result['generated_text']
            tokens = text.split()
            
            if len(tokens) > 0:
  
                unique_tokens = len(set(tokens))
                repetition_score = unique_tokens / len(tokens)
                repetition_scores.append(repetition_score)
                
         
                if len(tokens) >= 4:
                    bigrams = set(zip(tokens[:-1], tokens[1:]))
                    diversity_score = len(bigrams) / (len(tokens) - 1)
                    diversity_scores.append(diversity_score)
        
        return {
            'avg_repetition_score': np.mean(repetition_scores) if repetition_scores else 0,
            'avg_diversity_score': np.mean(diversity_scores) if diversity_scores else 0,
            'repetition_scores': repetition_scores,
            'diversity_scores': diversity_scores
        }


def create_inference_engine(model_path: str, **kwargs) -> SLMInference:
    """Factory function to create inference engine."""
    return SLMInference(model_path, **kwargs)


def run_evaluation_suite(
    model_path: str,
    test_data_path: str = None,
    test_prompts: List[str] = None
) -> Dict[str, any]:
    """
    Run complete evaluation suite.
    
    Args:
        model_path: Path to trained model
        test_data_path: Path to test data for perplexity
        test_prompts: Custom test prompts
        
    Returns:
        Complete evaluation results
    """
    # Default test prompts
    if test_prompts is None:
        test_prompts = [
            "Once upon a time",
            "The weather today is",
            "In the future, technology will",
            "The most important thing in life is",
            "Yesterday I went to"
        ]
    

    inference = create_inference_engine(model_path)
    evaluator = SLMEvaluator(inference)
    
    results = {
        'model_path': model_path,
        'model_config': inference.config.__dict__
    }

    if test_data_path and Path(test_data_path).exists():
        logger.info("Running perplexity evaluation...")
        results['perplexity'] = evaluator.perplexity_evaluation(test_data_path)
    
 
    logger.info("Running prompt evaluation...")
    results['prompt_evaluation'] = evaluator.prompt_evaluation(test_prompts)
    

    logger.info("Running coherence evaluation...")
    results['coherence'] = evaluator.coherence_evaluation(test_prompts)
    
    return results


# CLI for inference and evaluation
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SLM Inference and Evaluation")
    parser.add_argument("model_path", help="Path to trained model")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation suite")
    parser.add_argument("--test-data", help="Path to test data for perplexity")
    parser.add_argument("--prompt", help="Single prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation")
    parser.add_argument("--top-k", type=int, help="Top-k for generation")
    
    args = parser.parse_args()
    

    logging.basicConfig(level=logging.INFO)

    inference = create_inference_engine(args.model_path)
    
    if args.interactive:
        inference.interactive_generation()
    elif args.evaluate:
        results = run_evaluation_suite(args.model_path, args.test_data)
        print(json.dumps(results, indent=2, default=str))
    elif args.prompt:
        config = InferenceConfig(
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
        result = inference.generate_text(args.prompt, config)
        print(f"Prompt: {result['prompt']}")
        print(f"Generated: {result['generated_text']}")
    else:
        print("Specify --interactive, --evaluate, or --prompt")