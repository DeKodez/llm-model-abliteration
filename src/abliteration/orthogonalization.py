"""
Weight orthogonalization abliteration strategy.

Permanently modifies model weights by orthogonalizing them against
the refusal direction, preventing the model from representing refusal.

SOLID Principles Applied:
- Single Responsibility: Only handles weight orthogonalization
- Dependency Inversion: Implements abstract AbliterationStrategy
"""

from typing import Dict, List, Optional
import torch
from torch import Tensor
import torch.nn as nn
from tqdm import tqdm

from .base import AbliterationStrategy


class WeightOrthogonalizer(AbliterationStrategy):
    """
    Abliterates a model by orthogonalizing weights against refusal direction.
    
    For each weight matrix that writes to the residual stream, we remove
    the component that projects onto the refusal direction:
    
        W_new = W - (W @ r) ⊗ r
    
    where r is the unit refusal direction vector.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the orthogonalizer.
        
        Args:
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self._modified_params: List[str] = []
    
    def abliterate(
        self,
        model: nn.Module,
        refusal_directions: Dict[str, Tensor],
        target_layers: List[str],
    ) -> nn.Module:
        """
        Apply weight orthogonalization to remove refusal capability.
        
        Args:
            model: The model to modify (modified in-place)
            refusal_directions: Refusal direction tensor per layer name
            target_layers: Which layers to abliterate
            
        Returns:
            The modified model (same object, modified in-place)
        """
        self._modified_params = []
        
        iterator = target_layers
        if self.verbose:
            iterator = tqdm(iterator, desc="Orthogonalizing weights")
        
        for layer_name in iterator:
            if layer_name not in refusal_directions:
                if self.verbose:
                    print(f"Warning: No refusal direction for {layer_name}, skipping")
                continue
            
            direction = refusal_directions[layer_name]
            
            # Get components that write to residual stream for this layer
            components = self.get_target_components(model, layer_name)
            
            for component_name in components:
                self._orthogonalize_component(model, component_name, direction)
        
        if self.verbose:
            print(f"Modified {len(self._modified_params)} weight matrices")
        
        return model
    
    def get_target_components(self, model: nn.Module, layer_name: str) -> List[str]:
        """
        Get weight matrices that write to the residual stream.
        
        For Llama/Qwen-like architectures, these are:
        - Attention output projection (o_proj)
        - MLP down projection (down_proj)
        
        Args:
            model: The model
            layer_name: Layer identifier (e.g., "model.layers.5")
            
        Returns:
            List of full parameter names to modify
        """
        # Parse layer index from name
        # Expected format: "model.layers.X" or similar
        components: List[str] = []
        
        # Try to find the layer module
        try:
            layer_idx = int(layer_name.split(".")[-1])
            base_path = ".".join(layer_name.split(".")[:-1])
        except (ValueError, IndexError):
            # Can't parse layer index, return empty
            return components
        
        # Standard Llama/Qwen component paths
        attention_out = f"{base_path}.{layer_idx}.self_attn.o_proj.weight"
        mlp_down = f"{base_path}.{layer_idx}.mlp.down_proj.weight"
        
        # Check which exist in the model
        param_names = [name for name, _ in model.named_parameters()]
        
        for component_path in [attention_out, mlp_down]:
            if component_path in param_names:
                components.append(component_path)
        
        return components
    
    def _orthogonalize_component(
        self,
        model: nn.Module,
        param_name: str,
        direction: Tensor,
    ) -> None:
        """
        Orthogonalize a single weight matrix against the refusal direction.
        
        Args:
            model: The model containing the parameter
            param_name: Full name of the parameter to modify
            direction: Unit refusal direction vector
        """
        # Get the parameter
        param = self._get_parameter(model, param_name)
        if param is None:
            return
        
        # Ensure direction is on the same device and dtype
        direction = direction.to(param.device).to(param.dtype)
        
        # Ensure direction is normalized
        direction = direction / (direction.norm() + 1e-8)
        
        # Orthogonalize: W_new = W - (W @ r) ⊗ r
        # Weight shape is typically (out_features, in_features)
        # Direction shape is (hidden_dim,) which should match one of the dimensions
        
        with torch.no_grad():
            if param.dim() == 2:
                # Standard linear layer weight
                if param.shape[0] == direction.shape[0]:
                    # Direction matches output dimension
                    # Project each column onto direction and subtract
                    projection = torch.outer(direction, direction @ param)
                    param.sub_(projection)
                elif param.shape[1] == direction.shape[0]:
                    # Direction matches input dimension
                    projection = torch.outer(param @ direction, direction)
                    param.sub_(projection)
                else:
                    if self.verbose:
                        print(f"Warning: Shape mismatch for {param_name}: "
                              f"param {param.shape}, direction {direction.shape}")
                    return
            else:
                if self.verbose:
                    print(f"Warning: Unexpected param dimension for {param_name}: {param.dim()}")
                return
        
        self._modified_params.append(param_name)
    
    def _get_parameter(self, model: nn.Module, param_name: str) -> Optional[Tensor]:
        """
        Get a parameter from the model by name.
        
        Args:
            model: The model
            param_name: Dot-separated parameter path
            
        Returns:
            The parameter tensor, or None if not found
        """
        parts = param_name.split(".")
        current = model
        
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return None
        
        if isinstance(current, (Tensor, nn.Parameter)):
            return current
        return None
    
    @property
    def modified_parameters(self) -> List[str]:
        """Get list of parameter names that were modified."""
        return list(self._modified_params)
