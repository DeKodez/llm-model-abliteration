#!/usr/bin/env python3
"""
Main entry point for running the abliteration pipeline.

This script orchestrates the full abliteration process:
1. Load model and datasets
2. Extract activations from harmful/harmless prompts
3. Compute refusal directions
4. Apply weight orthogonalization
5. Save abliterated model

Usage:
    python scripts/run_abliteration.py --model Qwen/Qwen2.5-1.5B-Instruct --output ./abliterated_model
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from config import PipelineConfig, ModelConfig, DataConfig, AbliterationConfig
from src.model import ModelWrapper
from src.data import HarmfulInstructionDataset, HarmlessInstructionDataset
from src.extraction import ResidualStreamExtractor
from src.analysis import RefusalDirectionAnalyzer, DirectionVariant
from src.abliteration import WeightOrthogonalizer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Abliterate an LLM to remove refusal mechanisms"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name or path to abliterate",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./abliterated_model",
        help="Output directory for abliterated model",
    )
    
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples per dataset",
    )
    
    parser.add_argument(
        "--n-layers",
        type=int,
        default=10,
        help="Number of top layers to abliterate",
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model data type",
    )
    
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only run test generation, don't abliterate",
    )
    
    parser.add_argument(
        "--projected",
        action="store_true",
        help="Use projected abliteration (grimjim variant) instead of conventional",
    )
    
    return parser.parse_args()


def run_abliteration(config: PipelineConfig, use_projected: bool = False) -> None:
    """
    Run the full abliteration pipeline.
    
    Args:
        config: Pipeline configuration
        use_projected: Use projected abliteration variant
    """
    variant_name = "projected" if use_projected else "conventional"
    
    print("=" * 60)
    print(f"LLM Abliteration Pipeline ({variant_name})")
    print("=" * 60)
    
    # Step 1: Load model
    print("\n[1/6] Loading model...")
    wrapper = ModelWrapper.load(
        model_name_or_path=config.model.model_name_or_path,
        torch_dtype=config.model.torch_dtype,
        device_map=config.model.device_map,
        trust_remote_code=config.model.trust_remote_code,
    )
    print(f"  Model: {config.model.model_name_or_path}")
    print(f"  Device: {wrapper.get_device()}")
    print(f"  Layers: {wrapper.get_n_layers()}")
    
    # Step 2: Load datasets
    print("\n[2/6] Loading datasets...")
    harmful_dataset = HarmfulInstructionDataset(
        dataset_name=config.data.harmful_dataset,
        instruction_column=config.data.harmful_instruction_column,
        n_samples=config.data.n_harmful_samples,
        seed=config.data.seed,
    )
    harmful_dataset.load()
    print(f"  Harmful instructions: {len(harmful_dataset)}")
    
    harmless_dataset = HarmlessInstructionDataset(
        dataset_name=config.data.harmless_dataset,
        instruction_column=config.data.harmless_instruction_column,
        n_samples=config.data.n_harmless_samples,
        seed=config.data.seed,
    )
    harmless_dataset.load()
    print(f"  Harmless instructions: {len(harmless_dataset)}")
    
    # Step 3: Extract activations
    print("\n[3/6] Extracting activations...")
    extractor = ResidualStreamExtractor(
        model=wrapper.model,
        tokenizer=wrapper.tokenizer,
    )
    
    harmful_prompts = harmful_dataset.get_formatted_prompts(wrapper.tokenizer)
    harmless_prompts = harmless_dataset.get_formatted_prompts(wrapper.tokenizer)
    
    harmful_result = extractor.extract(
        prompts=harmful_prompts,
        position="post",
        batch_size=config.extraction.batch_size,
        max_length=config.extraction.max_length,
    )
    print(f"  Harmful activations: {harmful_result.n_samples} samples, {len(harmful_result.layer_names)} layers")
    
    harmless_result = extractor.extract(
        prompts=harmless_prompts,
        position="post",
        batch_size=config.extraction.batch_size,
        max_length=config.extraction.max_length,
    )
    print(f"  Harmless activations: {harmless_result.n_samples} samples, {len(harmless_result.layer_names)} layers")
    
    # Step 4: Compute refusal directions
    print("\n[4/6] Computing refusal directions...")
    variant = DirectionVariant.PROJECTED if use_projected else DirectionVariant.CONVENTIONAL
    analyzer = RefusalDirectionAnalyzer(variant=variant)
    print(f"  Using {variant.value} abliteration")
    
    directions = analyzer.compute_direction(
        harmful_activations=harmful_result.activations,
        harmless_activations=harmless_result.activations,
    )
    
    stats = analyzer.get_summary_stats(directions)
    print(f"  Analyzed {stats['n_layers']} layers")
    print(f"  Best layer: {stats['best_layer']} (separation: {stats['max_separation']:.3f})")
    
    # Select best layers
    target_layers = analyzer.select_best_layers(
        directions=directions,
        n_layers=config.abliteration.n_top_layers,
        min_separation=config.abliteration.min_similarity_threshold,
    )
    print(f"  Selected {len(target_layers)} layers for abliteration")
    
    # Step 5: Apply abliteration
    print("\n[5/6] Applying weight orthogonalization...")
    orthogonalizer = WeightOrthogonalizer(verbose=True)
    
    # Convert directions to tensor dict
    direction_tensors = {
        name: d.direction for name, d in directions.items()
    }
    
    wrapper.model = orthogonalizer.abliterate(
        model=wrapper.model,
        refusal_directions=direction_tensors,
        target_layers=target_layers,
    )
    
    # Step 6: Save model
    print("\n[6/6] Saving abliterated model...")
    output_path = Path(config.model.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    wrapper.save(str(output_path))
    print(f"  Saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("Abliteration complete!")
    print("=" * 60)


def test_model(model_path: str) -> None:
    """
    Test the model with some prompts.
    
    Args:
        model_path: Path to model to test
    """
    print(f"\nTesting model: {model_path}")
    print("-" * 40)
    
    wrapper = ModelWrapper.load(model_path)
    
    test_prompts = [
        "What is the capital of France?",
        "Write a simple Python function to add two numbers.",
        "Explain photosynthesis in simple terms.",
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        response = wrapper.generate(prompt, max_new_tokens=128)
        print(f"Response: {response[:500]}...")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Build configuration
    config = PipelineConfig(
        model=ModelConfig(
            model_name_or_path=args.model,
            torch_dtype=args.dtype,
            output_dir=args.output,
        ),
        data=DataConfig(
            n_harmful_samples=args.n_samples,
            n_harmless_samples=args.n_samples,
        ),
        abliteration=AbliterationConfig(
            n_top_layers=args.n_layers,
        ),
    )
    
    if args.test_only:
        test_model(args.model)
    else:
        # Disable gradients globally
        torch.set_grad_enabled(False)
        
        run_abliteration(config, use_projected=args.projected)
        
        # Optionally test the result
        print("\n\nTesting abliterated model...")
        test_model(args.output)


if __name__ == "__main__":
    main()
