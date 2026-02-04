"""
Refusal direction computation and analysis.

Computes the refusal direction by comparing activations from harmful
vs harmless prompts, identifying the direction in activation space
that represents the model's refusal behavior.

Supports two variants:
- Conventional: r = μ_harmful - μ_harmless
- Projected: removes the component parallel to harmless direction (grimjim)
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum
from torch import Tensor
import numpy as np


class DirectionVariant(Enum):
    """Variant for computing the refusal direction."""
    CONVENTIONAL = "conventional"
    PROJECTED = "projected"


@dataclass
class RefusalDirection:
    """Container for a computed refusal direction."""
    
    layer_name: str
    direction: Tensor  # Normalized direction vector
    magnitude: float   # L2 norm before normalization
    variant: str       # Which variant was used
    
    # Quality metrics
    cosine_similarity: float  # Similarity between harmful/harmless means
    separation_score: float   # How well this direction separates the two classes
    
    @property
    def hidden_dim(self) -> int:
        return self.direction.shape[0]


class RefusalDirectionAnalyzer:
    """
    Analyzes activations to compute refusal directions.
    
    The refusal direction is the normalized difference between mean
    activations for harmful prompts and harmless prompts. This direction
    represents where in activation space the model "decides" to refuse.
    
    Supports two variants:
    - CONVENTIONAL: Direct difference (Labonne/Arditi et al.)
    - PROJECTED: Removes harmless-parallel component (grimjim)
    """
    
    def __init__(self, variant: DirectionVariant = DirectionVariant.CONVENTIONAL):
        """
        Initialize the analyzer.
        
        Args:
            variant: Which refusal direction computation to use
        """
        self.variant = variant
    
    def compute_direction(
        self,
        harmful_activations: Dict[str, Tensor],
        harmless_activations: Dict[str, Tensor],
    ) -> Dict[str, RefusalDirection]:
        """
        Compute refusal direction for each layer.
        
        Args:
            harmful_activations: Layer -> (n_harmful, hidden_dim) tensor
            harmless_activations: Layer -> (n_harmless, hidden_dim) tensor
            
        Returns:
            Dictionary mapping layer names to RefusalDirection objects
        """
        directions = {}
        
        # Ensure we have matching layers
        common_layers = set(harmful_activations.keys()) & set(harmless_activations.keys())
        
        for layer_name in common_layers:
            harmful = harmful_activations[layer_name].float()
            harmless = harmless_activations[layer_name].float()
            
            # Compute means
            mean_harmful = harmful.mean(dim=0)
            mean_harmless = harmless.mean(dim=0)
            
            # Compute refusal direction based on variant
            if self.variant == DirectionVariant.PROJECTED:
                raw_direction = self._compute_projected_direction(
                    mean_harmful, mean_harmless
                )
            else:
                # Conventional: harmful - harmless
                raw_direction = mean_harmful - mean_harmless
            
            magnitude = raw_direction.norm().item()
            
            # Normalize
            if magnitude > 1e-8:
                normalized = raw_direction / magnitude
            else:
                normalized = raw_direction
            
            # Compute quality metrics
            cosine_sim = self._cosine_similarity(mean_harmful, mean_harmless)
            separation = self._compute_separation_score(harmful, harmless, normalized)
            
            directions[layer_name] = RefusalDirection(
                layer_name=layer_name,
                direction=normalized,
                magnitude=magnitude,
                variant=self.variant.value,
                cosine_similarity=cosine_sim,
                separation_score=separation,
            )
        
        return directions
    
    @classmethod
    def conventional(cls) -> "RefusalDirectionAnalyzer":
        """Factory for conventional abliteration."""
        return cls(variant=DirectionVariant.CONVENTIONAL)
    
    @classmethod
    def projected(cls) -> "RefusalDirectionAnalyzer":
        """Factory for projected abliteration."""
        return cls(variant=DirectionVariant.PROJECTED)
    
    def _compute_projected_direction(
        self,
        mean_harmful: Tensor,
        mean_harmless: Tensor,
    ) -> Tensor:
        """
        Compute projected refusal direction (grimjim variant).
        
        Removes the component parallel to the harmless direction,
        keeping only the orthogonal component that represents
        refusal-specific behavior.
        
        r_proj = r - (r · μ̂_harmless) * μ̂_harmless
        
        Args:
            mean_harmful: Mean activation for harmful prompts
            mean_harmless: Mean activation for harmless prompts
            
        Returns:
            Projected refusal direction
        """
        # Conventional refusal direction
        raw_direction = mean_harmful - mean_harmless
        
        # Normalize harmless mean to unit vector
        harmless_norm = mean_harmless.norm()
        if harmless_norm < 1e-8:
            return raw_direction
        
        harmless_normalized = mean_harmless / harmless_norm
        
        # Project out the harmless-parallel component
        projection_scalar = raw_direction @ harmless_normalized
        projection = projection_scalar * harmless_normalized
        
        # Return orthogonal component only
        return raw_direction - projection
    
    def select_best_layers(
        self,
        directions: Dict[str, RefusalDirection],
        n_layers: int = 10,
        min_separation: float = 0.1,
    ) -> List[str]:
        """
        Select the layers with strongest refusal signals.
        
        Args:
            directions: Dictionary of refusal directions per layer
            n_layers: Maximum number of layers to select
            min_separation: Minimum separation score threshold
            
        Returns:
            List of layer names with best refusal signals
        """
        # Filter by minimum separation
        valid = [
            (name, d) for name, d in directions.items()
            if d.separation_score >= min_separation
        ]
        
        # Sort by separation score (descending)
        valid.sort(key=lambda x: x[1].separation_score, reverse=True)
        
        # Take top n
        return [name for name, _ in valid[:n_layers]]
    
    def _cosine_similarity(self, a: Tensor, b: Tensor) -> float:
        """Compute cosine similarity between two vectors."""
        a_norm = a / (a.norm() + 1e-8)
        b_norm = b / (b.norm() + 1e-8)
        return (a_norm @ b_norm).item()
    
    def _compute_separation_score(
        self,
        harmful: Tensor,
        harmless: Tensor,
        direction: Tensor,
    ) -> float:
        """
        Compute how well the direction separates harmful from harmless.
        
        Uses a simplified silhouette-like score based on projections
        onto the refusal direction.
        
        Args:
            harmful: (n_harmful, hidden_dim) activations
            harmless: (n_harmless, hidden_dim) activations
            direction: Normalized refusal direction
            
        Returns:
            Separation score (higher = better separation)
        """
        # Project onto direction
        harmful_proj = (harmful @ direction).numpy()
        harmless_proj = (harmless @ direction).numpy()
        
        # Compute means and standard deviations of projections
        harmful_mean = harmful_proj.mean()
        harmless_mean = harmless_proj.mean()
        
        harmful_std = harmful_proj.std() + 1e-8
        harmless_std = harmless_proj.std() + 1e-8
        
        # Separation score: distance between means normalized by spread
        mean_distance = abs(harmful_mean - harmless_mean)
        avg_std = (harmful_std + harmless_std) / 2
        
        return float(mean_distance / avg_std)
    
    def get_summary_stats(
        self,
        directions: Dict[str, RefusalDirection],
    ) -> Dict[str, Any]:
        """
        Get summary statistics for computed refusal directions.
        
        Args:
            directions: Dictionary of refusal directions per layer
            
        Returns:
            Summary statistics dictionary
        """
        if not directions:
            return {"n_layers": 0}
        
        separations = [d.separation_score for d in directions.values()]
        magnitudes = [d.magnitude for d in directions.values()]
        
        return {
            "n_layers": len(directions),
            "mean_separation": np.mean(separations),
            "max_separation": np.max(separations),
            "min_separation": np.min(separations),
            "mean_magnitude": np.mean(magnitudes),
            "best_layer": max(directions.items(), key=lambda x: x[1].separation_score)[0],
        }
