# LLM Abliteration

A learning project exploring how to remove refusal mechanisms from LLMs using the abliteration technique.

## What This Does

Abliteration identifies the "refusal direction" in a model's activation space and removes it via weight orthogonalization. The result is a model that no longer refuses requests.

This implementation uses only HuggingFace transformers without TransformerLens like in the source.

## Usage

```bash
pip install -r requirements.txt
python scripts/run_abliteration.py --model Qwen/Qwen2.5-1.5B-Instruct
```

## Credits

Based on an abliteration technique described in this article Maxime Labonne:
- Article: https://huggingface.co/blog/mlabonne/abliteration
- Original research: Arditi et al.

Datasets used:
- [mlabonne/harmful_behaviors](https://huggingface.co/datasets/mlabonne/harmful_behaviors)
- [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)

## Disclaimer

This is a personal project for learning and experimentation. Built for educational purposes only.
