"""
Model loading and saving wrapper.

Provides a unified interface for model I/O operations.
"""

from typing import Any
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelWrapper:
    """
    Wrapper for loading and saving HuggingFace models.
    
    Provides convenient methods for model I/O with proper
    configuration handling.
    """
    
    model: Any = None
    tokenizer: Any = None
    model_name: str = ""
    
    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        trust_remote_code: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ) -> "ModelWrapper":
        """
        Load a model and tokenizer from HuggingFace Hub or local path.
        
        Args:
            model_name_or_path: Model identifier or path
            torch_dtype: Data type for model weights
            device_map: Device placement strategy
            trust_remote_code: Whether to trust remote code
            load_in_8bit: Load in 8-bit quantization
            load_in_4bit: Load in 4-bit quantization
            
        Returns:
            ModelWrapper instance with loaded model and tokenizer
        """
        # Parse dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(torch_dtype, torch.bfloat16)
        
        # Build model kwargs
        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }
        
        # Add quantization if requested
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        
        # Ensure padding token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )
        
        # Disable gradient computation (we're not training)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        return cls(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name_or_path,
        )
    
    def save(
        self,
        output_dir: str,
        save_tokenizer: bool = True,
    ) -> None:
        """
        Save the model and optionally tokenizer to disk.
        
        Args:
            output_dir: Directory to save to
            save_tokenizer: Whether to also save the tokenizer
        """
        self.model.save_pretrained(output_dir)
        
        if save_tokenizer:
            self.tokenizer.save_pretrained(output_dir)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt string
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text (excluding the prompt)
        """
        # Format as chat if possible
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted = prompt
        
        # Tokenize
        inputs = self.tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )
        
        # Decode only the new tokens
        new_tokens = outputs[0][input_length:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    def get_device(self) -> str:
        """Get the device the model is on."""
        try:
            param = next(self.model.parameters())
            return str(param.device)
        except StopIteration:
            return "cpu"
    
    def get_n_layers(self) -> int:
        """Get the number of transformer layers."""
        if hasattr(self.model, "config"):
            if hasattr(self.model.config, "num_hidden_layers"):
                return self.model.config.num_hidden_layers
            if hasattr(self.model.config, "n_layer"):
                return self.model.config.n_layer
        
        # Fallback: count layers
        count = 0
        for name, _ in self.model.named_modules():
            if ".layers." in name:
                try:
                    idx = int(name.split(".layers.")[-1].split(".")[0])
                    count = max(count, idx + 1)
                except ValueError:
                    pass
        return count
