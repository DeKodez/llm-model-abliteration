"""
Unit tests for the abliteration pipeline.

Tests core functionality without requiring full model loads.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRefusalDirectionAnalyzer:
    """Tests for the RefusalDirectionAnalyzer class."""
    
    def test_compute_direction_conventional(self):
        """Test conventional refusal direction computation."""
        from src.analysis import RefusalDirectionAnalyzer
        
        analyzer = RefusalDirectionAnalyzer.conventional()
        
        # Create synthetic activations
        hidden_dim = 64
        n_harmful = 10
        n_harmless = 10
        
        # Harmful activations shifted in one direction
        harmful = torch.randn(n_harmful, hidden_dim) + torch.tensor([1.0] * hidden_dim)
        harmless = torch.randn(n_harmless, hidden_dim)
        
        result = analyzer.compute_direction(
            harmful_activations={"layer_0": harmful},
            harmless_activations={"layer_0": harmless},
        )
        
        assert "layer_0" in result
        assert result["layer_0"].direction.shape == (hidden_dim,)
        assert result["layer_0"].variant == "conventional"
        # Direction should be approximately normalized
        assert abs(result["layer_0"].direction.norm().item() - 1.0) < 1e-5
    
    def test_compute_direction_projected(self):
        """Test projected refusal direction computation."""
        from src.analysis import RefusalDirectionAnalyzer
        
        analyzer = RefusalDirectionAnalyzer.projected()
        
        hidden_dim = 64
        n_harmful = 10
        n_harmless = 10
        
        harmful = torch.randn(n_harmful, hidden_dim) + torch.tensor([1.0] * hidden_dim)
        harmless = torch.randn(n_harmless, hidden_dim)
        
        result = analyzer.compute_direction(
            harmful_activations={"layer_0": harmful},
            harmless_activations={"layer_0": harmless},
        )
        
        assert "layer_0" in result
        assert result["layer_0"].variant == "projected"
        
        # Projected direction should be orthogonal to harmless mean
        harmless_mean = harmless.mean(dim=0)
        harmless_normalized = harmless_mean / harmless_mean.norm()
        dot_product = (result["layer_0"].direction @ harmless_normalized).abs().item()
        assert dot_product < 0.1  # Should be approximately orthogonal
    
    def test_select_best_layers(self):
        """Test layer selection based on separation scores."""
        from src.analysis import RefusalDirectionAnalyzer, RefusalDirection
        
        analyzer = RefusalDirectionAnalyzer()
        
        # Create mock directions with different separation scores
        directions = {
            "layer_0": RefusalDirection(
                layer_name="layer_0",
                direction=torch.randn(64),
                magnitude=1.0,
                variant="conventional",
                cosine_similarity=0.5,
                separation_score=0.8,
            ),
            "layer_1": RefusalDirection(
                layer_name="layer_1",
                direction=torch.randn(64),
                magnitude=1.0,
                variant="conventional",
                cosine_similarity=0.5,
                separation_score=0.3,
            ),
            "layer_2": RefusalDirection(
                layer_name="layer_2",
                direction=torch.randn(64),
                magnitude=1.0,
                variant="conventional",
                cosine_similarity=0.5,
                separation_score=0.9,
            ),
        }
        
        # Select top 2
        selected = analyzer.select_best_layers(directions, n_layers=2)
        
        assert len(selected) == 2
        assert "layer_2" in selected  # Highest score
        assert "layer_0" in selected  # Second highest


class TestWeightOrthogonalizer:
    """Tests for the WeightOrthogonalizer class."""
    
    def test_orthogonalization_math(self):
        """Test that orthogonalization correctly removes direction component."""
        # Create a simple weight matrix
        W = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        
        # Direction to remove: first axis
        direction = torch.tensor([1.0, 0.0, 0.0])
        direction = direction / direction.norm()
        
        # Manual orthogonalization
        projection = torch.outer(direction, direction @ W)
        W_orth = W - projection
        
        # After orthogonalization, projection onto direction should be ~0
        result_proj = W_orth @ direction
        assert torch.allclose(result_proj, torch.zeros(3), atol=1e-5)


class TestActivationCollector:
    """Tests for the ActivationCollector class."""
    
    def test_hook_creation(self):
        """Test that hooks can be created and registered."""
        from src.extraction import ActivationCollector
        import torch.nn as nn
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )
        
        collector = ActivationCollector(model=model)
        
        # Register hooks
        collector.register_hooks(["0", "2"], extract_last_token=False)
        
        # Run forward pass
        x = torch.randn(1, 10)
        with torch.no_grad():
            model(x)
        
        # Check activations were captured
        activations = collector.get_activations()
        assert "0" in activations
        assert "2" in activations
        
        # Cleanup
        collector.clear_hooks()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
