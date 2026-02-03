"""
Residual stream extraction from transformer models.

Extracts activations at different positions in the residual stream
for refusal direction analysis.

SOLID Principles Applied:
- Single Responsibility: Only handles residual stream extraction logic
- Dependency Inversion: Depends on ActivationCollector abstraction
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import torch
from torch import Tensor
from tqdm import tqdm

from .hooks import ActivationCollector, get_layer_names_for_model


@dataclass
class ExtractionResult:
    """Container for extracted activations."""
    
    # Activations per layer: layer_name -> (n_samples, hidden_dim)
    activations: Dict[str, Tensor]
    
    # Metadata
    n_samples: int
    layer_names: List[str]
    position: str  # "pre", "mid", or "post"


class ResidualStreamExtractor:
    """
    Extracts residual stream activations from a transformer model.
    
    This class orchestrates the activation collection process,
    running inference on a set of prompts and collecting activations
    at specified residual stream positions.
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: Optional[str] = None,
    ):
        """
        Initialize the extractor.
        
        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            device: Device to run inference on (auto-detected if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or self._detect_device()
        self.collector = ActivationCollector(model=model)
    
    def _detect_device(self) -> str:
        """Detect the device the model is on."""
        try:
            # Try to get device from model parameters
            param = next(self.model.parameters())
            return str(param.device)
        except StopIteration:
            return "cpu"
    
    def extract(
        self,
        prompts: List[str],
        position: str = "post",
        target_layers: Optional[List[int]] = None,
        batch_size: int = 1,
        max_length: int = 512,
        show_progress: bool = True,
    ) -> ExtractionResult:
        """
        Extract residual stream activations for a list of prompts.
        
        Args:
            prompts: List of formatted prompt strings
            position: Residual stream position ("pre", "mid", "post")
            target_layers: Specific layer indices to extract (None = all)
            batch_size: Batch size for inference
            max_length: Maximum sequence length
            show_progress: Show progress bar
            
        Returns:
            ExtractionResult with activations per layer
        """
        # Get layer names for this model architecture
        layer_map = get_layer_names_for_model(self.model)
        
        if position not in layer_map:
            raise ValueError(f"Invalid position '{position}'. Must be one of: {list(layer_map.keys())}")
        
        layer_names = layer_map[position]
        
        # Filter to target layers if specified
        if target_layers is not None:
            layer_names = [layer_names[i] for i in target_layers if i < len(layer_names)]
        
        # Register hooks
        self.collector.register_hooks(layer_names, extract_last_token=True)
        
        # Storage for all activations
        all_activations: Dict[str, List[Tensor]] = {name: [] for name in layer_names}
        
        # Process prompts
        iterator = range(0, len(prompts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Extracting {position} activations")
        
        try:
            for i in iterator:
                batch_prompts = prompts[i:i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Clear previous activations
                self.collector.clear()
                
                # Forward pass
                with torch.no_grad():
                    self.model(**inputs)
                
                # Collect activations
                activations = self.collector.get_activations()
                for name in layer_names:
                    if name in activations:
                        all_activations[name].append(activations[name])
        finally:
            # Clean up hooks
            self.collector.clear_hooks()
        
        # Stack activations: list of (batch, hidden) -> (n_samples, hidden)
        stacked = {}
        for name, act_list in all_activations.items():
            if act_list:
                stacked[name] = torch.cat(act_list, dim=0)
        
        return ExtractionResult(
            activations=stacked,
            n_samples=len(prompts),
            layer_names=layer_names,
            position=position,
        )
    
    def extract_all_positions(
        self,
        prompts: List[str],
        target_layers: Optional[List[int]] = None,
        batch_size: int = 1,
        max_length: int = 512,
        show_progress: bool = True,
    ) -> Dict[str, ExtractionResult]:
        """
        Extract activations from all residual stream positions.
        
        Args:
            prompts: List of formatted prompt strings
            target_layers: Specific layer indices to extract (None = all)
            batch_size: Batch size for inference
            max_length: Maximum sequence length
            show_progress: Show progress bar
            
        Returns:
            Dictionary mapping position names to ExtractionResults
        """
        results = {}
        for position in ["pre", "mid", "post"]:
            results[position] = self.extract(
                prompts=prompts,
                position=position,
                target_layers=target_layers,
                batch_size=batch_size,
                max_length=max_length,
                show_progress=show_progress,
            )
        return results
