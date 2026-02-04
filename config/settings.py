"""
Configuration dataclasses for the abliteration pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class ResidualStreamPosition(Enum):
    """Position in the residual stream to extract activations from."""
    PRE = "pre"    # Start of block (input)
    MID = "mid"    # Between attention and MLP
    POST = "post"  # After MLP (output)


class AbliterationMethod(Enum):
    """Available abliteration strategies."""
    WEIGHT_ORTHOGONALIZATION = "weight_orthogonalization"
    INFERENCE_TIME = "inference_time"


class RefusalDirectionVariant(Enum):
    """Variant for computing the refusal direction."""
    CONVENTIONAL = "conventional"  # r = μ_harmful - μ_harmless
    PROJECTED = "projected"        # r = r - (r · μ̂_harmless) * μ̂_harmless


@dataclass
class ModelConfig:
    """Configuration for model loading and saving."""
    
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    torch_dtype: str = "bfloat16"  # bfloat16, float16, float32
    device_map: str = "auto"
    trust_remote_code: bool = True
    output_dir: str = "./abliterated_model"
    
    # Quantization options (optional, reduces VRAM)
    load_in_8bit: bool = False
    load_in_4bit: bool = False


@dataclass
class DataConfig:
    """Configuration for instruction datasets."""
    
    harmful_dataset: str = "mlabonne/harmful_behaviors"
    harmless_dataset: str = "tatsu-lab/alpaca"
    
    # Number of samples to use (None = all)
    n_harmful_samples: Optional[int] = 100
    n_harmless_samples: Optional[int] = 100
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Instruction column names (varies by dataset)
    harmful_instruction_column: str = "text"
    harmless_instruction_column: str = "instruction"


@dataclass
class ExtractionConfig:
    """Configuration for activation extraction."""
    
    # Which residual stream positions to extract
    positions: List[ResidualStreamPosition] = field(
        default_factory=lambda: [ResidualStreamPosition.POST]
    )
    
    # Which layers to extract from (None = all layers)
    target_layers: Optional[List[int]] = None
    
    # Batch size for inference
    batch_size: int = 1
    
    # Maximum sequence length
    max_length: int = 512


@dataclass
class AbliterationConfig:
    """Configuration for the abliteration process."""
    
    method: AbliterationMethod = AbliterationMethod.WEIGHT_ORTHOGONALIZATION
    
    # Refusal direction variant: conventional or projected
    direction_variant: RefusalDirectionVariant = RefusalDirectionVariant.CONVENTIONAL
    
    # Layers to apply abliteration to (None = auto-select best)
    target_layers: Optional[List[int]] = None
    
    # If True, automatically select layers with strongest refusal signal
    auto_select_layers: bool = True
    
    # Number of top layers to select when auto-selecting
    n_top_layers: int = 10
    
    # Minimum cosine similarity threshold for valid refusal direction
    min_similarity_threshold: float = 0.1


@dataclass
class PipelineConfig:
    """Master configuration combining all sub-configs."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    abliteration: AbliterationConfig = field(default_factory=AbliterationConfig)
    
    # Logging
    verbose: bool = True
    
    @classmethod
    def for_qwen_1_5b(cls) -> "PipelineConfig":
        """Factory method for Qwen2.5-1.5B-Instruct."""
        return cls(
            model=ModelConfig(
                model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
                torch_dtype="bfloat16",
            )
        )
    
    @classmethod
    def for_llama_1b(cls) -> "PipelineConfig":
        """Factory method for Llama-3.2-1B-Instruct."""
        return cls(
            model=ModelConfig(
                model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
                torch_dtype="bfloat16",
            )
        )
