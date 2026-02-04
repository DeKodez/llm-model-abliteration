# LLM Abliteration

A learning project exploring how to remove refusal mechanisms from LLMs using abliteration techniques.

## What This Does

Abliteration identifies the "refusal direction" in a model's activation space and removes it via weight orthogonalization. The result is a model that no longer refuses requests.

This implementation uses only HuggingFace transformers without TransformerLens like in the source.

## Abliteration Variants

**Conventional** (default): Uses the direct difference between harmful and harmless activation means.
```
r = μ_harmful - μ_harmless
```

**Projected**: Removes the component parallel to the harmless direction, preserving model helpfulness.
```
r_proj = r - (r · μ̂_harmless) * μ̂_harmless
```

## Usage

```bash
pip install -r requirements.txt

# Conventional abliteration
python scripts/run_abliteration.py --model Qwen/Qwen2.5-1.5B-Instruct

# Projected abliteration
python scripts/run_abliteration.py --model Qwen/Qwen2.5-1.5B-Instruct --projected
```

## Credits

Conventional abliteration:
- Maxime Labonne: https://huggingface.co/blog/mlabonne/abliteration
- Original research: Arditi et al., "Refusal in LLMs is mediated by a single direction"

Projected abliteration:
- Jim Lai (grimjim): https://huggingface.co/blog/grimjim/projected-abliteration

Datasets used:
- [mlabonne/harmful_behaviors](https://huggingface.co/datasets/mlabonne/harmful_behaviors)
- [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)

## Disclaimer

This is a personal project for learning and experimentation. Built for educational purposes only.
