"""
Abstract base class for abliteration strategies.

SOLID Principles Applied:
- Open/Closed: New abliteration methods can be added by subclassing
- Liskov Substitution: All strategies are interchangeable
- Interface Segregation: Minimal required interface
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
from torch import Tensor


class AbliterationStrategy(ABC):
    """
    Abstract base class for abliteration strategies.
    
    Abliteration modifies a model to remove its refusal capability
    by eliminating the refusal direction from relevant components.
    """
    
    @abstractmethod
    def abliterate(
        self,
        model: Any,
        refusal_directions: Dict[str, Tensor],
        target_layers: List[str],
    ) -> Any:
        """
        Apply abliteration to the model.
        
        Args:
            model: The model to modify
            refusal_directions: Refusal direction per layer
            target_layers: Which layers to apply abliteration to
            
        Returns:
            Modified model (may be in-place)
        """
        ...
    
    @abstractmethod
    def get_target_components(self, model: Any, layer_name: str) -> List[str]:
        """
        Get the weight matrices to modify for a given layer.
        
        Args:
            model: The model
            layer_name: Name of the layer
            
        Returns:
            List of parameter names to orthogonalize
        """
        ...
