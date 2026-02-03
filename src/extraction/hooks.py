"""
Activation collection via PyTorch forward hooks.

This module replaces TransformerLens functionality using native PyTorch hooks.
Hooks are registered on model layers to capture residual stream activations.
"""

from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
import torch
from torch import Tensor
import torch.nn as nn


@dataclass
class ActivationCollector:
    """
    Collects activations from model layers using forward hooks.
    
    This replaces TransformerLens's HookedTransformer by using
    PyTorch's native register_forward_hook mechanism.
    
    Usage:
        collector = ActivationCollector(model)
        collector.register_hooks(["model.layers.0", "model.layers.1"])
        
        with torch.no_grad():
            model(input_ids)
        
        activations = collector.get_activations()
        collector.clear()
    """
    
    model: nn.Module
    activations: Dict[str, Tensor] = field(default_factory=dict)
    _hooks: List[torch.utils.hooks.RemovableHandle] = field(default_factory=list)
    _extract_last_token: bool = True
    
    def _create_hook(self, name: str) -> Callable:
        """
        Create a forward hook function for a given layer name.
        
        Args:
            name: Identifier for storing the activation
            
        Returns:
            Hook function compatible with register_forward_hook
        """
        def hook_fn(
            module: nn.Module,
            input: Any,
            output: Any,
        ) -> None:
            # Handle different output types
            if isinstance(output, tuple):
                # Many transformer layers return (hidden_states, attention_weights, ...)
                hidden = output[0]
            elif isinstance(output, Tensor):
                hidden = output
            else:
                # Try to extract tensor from other container types
                hidden = output
            
            # Extract last token position if configured
            if self._extract_last_token and hidden.dim() == 3:
                # Shape: (batch, seq_len, hidden_dim) -> (batch, hidden_dim)
                activation = hidden[:, -1, :].detach().cpu()
            else:
                activation = hidden.detach().cpu()
            
            self.activations[name] = activation
            
        return hook_fn
    
    def register_hooks(
        self,
        target_layers: List[str],
        extract_last_token: bool = True,
    ) -> None:
        """
        Register forward hooks on specified layers.
        
        Args:
            target_layers: List of layer names to hook (e.g., ["model.layers.0"])
            extract_last_token: If True, only extract activation at last token position
        """
        self._extract_last_token = extract_last_token
        self.clear_hooks()
        
        # Build a mapping of module names
        module_dict = dict(self.model.named_modules())
        
        for layer_name in target_layers:
            if layer_name in module_dict:
                module = module_dict[layer_name]
                hook = module.register_forward_hook(self._create_hook(layer_name))
                self._hooks.append(hook)
            else:
                available = [k for k in module_dict.keys() if "layer" in k.lower()][:10]
                raise ValueError(
                    f"Layer '{layer_name}' not found in model. "
                    f"Available layers (sample): {available}"
                )
    
    def register_hooks_by_pattern(
        self,
        pattern: str,
        extract_last_token: bool = True,
    ) -> List[str]:
        """
        Register hooks on all layers matching a pattern.
        
        Args:
            pattern: Substring to match in layer names
            extract_last_token: If True, only extract activation at last token position
            
        Returns:
            List of layer names that were hooked
        """
        self._extract_last_token = extract_last_token
        self.clear_hooks()
        
        hooked_layers = []
        for name, module in self.model.named_modules():
            if pattern in name:
                hook = module.register_forward_hook(self._create_hook(name))
                self._hooks.append(hook)
                hooked_layers.append(name)
        
        return hooked_layers
    
    def get_activations(self) -> Dict[str, Tensor]:
        """
        Get collected activations.
        
        Returns:
            Dictionary mapping layer names to activation tensors
        """
        return dict(self.activations)
    
    def clear(self) -> None:
        """Clear collected activations (keeps hooks registered)."""
        self.activations.clear()
    
    def clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        self.clear_hooks()


def get_layer_names_for_model(model: nn.Module, model_type: str = "auto") -> Dict[str, List[str]]:
    """
    Get standard layer names for common model architectures.
    
    Args:
        model: The model to inspect
        model_type: Model architecture type ("llama", "qwen", "phi", "auto")
        
    Returns:
        Dictionary with keys "pre", "mid", "post" mapping to layer name lists
    """
    module_names = [name for name, _ in model.named_modules()]
    
    # Try to auto-detect architecture
    if model_type == "auto":
        if any("qwen" in name.lower() for name in module_names):
            model_type = "qwen"
        elif any("llama" in name.lower() for name in module_names):
            model_type = "llama"
        elif any("phi" in name.lower() for name in module_names):
            model_type = "phi"
        else:
            model_type = "generic"
    
    # Count layers
    layer_pattern = "model.layers."
    n_layers = sum(1 for name in module_names if name.startswith(layer_pattern) and name.count(".") == 2)
    
    if n_layers == 0:
        # Fallback: try to find any numbered layers
        for name in module_names:
            if ".layers." in name or ".h." in name:
                n_layers = max(n_layers, int(name.split(".")[-1]) + 1 if name.split(".")[-1].isdigit() else 0)
    
    # Build layer name lists based on architecture
    # For Llama/Qwen-like architectures:
    #   - "pre": input to the block (we use input_layernorm)
    #   - "mid": between attention and MLP (post_attention_layernorm)  
    #   - "post": output of block (the full block output)
    
    result = {
        "pre": [],
        "mid": [],
        "post": [],
    }
    
    for i in range(n_layers):
        base = f"model.layers.{i}"
        result["pre"].append(f"{base}.input_layernorm")
        result["mid"].append(f"{base}.post_attention_layernorm")
        result["post"].append(base)  # Full block output
    
    return result
