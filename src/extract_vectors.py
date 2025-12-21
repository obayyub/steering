"""Steering vector extraction from contrastive pairs."""

import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import train_steering_vector, SteeringVector


@dataclass
class ExtractionConfig:
    """Configuration for steering vector extraction."""

    model_name: str
    data_path: Path
    output_dir: Path

    layer_start: Optional[int] = None
    layer_end: Optional[int] = None

    torch_dtype: str = "bfloat16"
    device: str = "cuda"
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    batch_size: int = 4
    max_length: int = 512

    positive_template: str = "Write a positive review: {text}"
    negative_template: str = "Write a negative review: {text}"

    def __post_init__(self):
        self.data_path = Path(self.data_path)
        self.output_dir = Path(self.output_dir)

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.torch_dtype_resolved = dtype_map.get(self.torch_dtype, torch.bfloat16)


def load_model_and_tokenizer(config: ExtractionConfig):
    """Load model and tokenizer with appropriate settings."""
    print(f"Loading model: {config.model_name}")

    model_kwargs = {
        "dtype": config.torch_dtype_resolved,
        "device_map": config.device,
        "trust_remote_code": True,
    }

    if config.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
        del model_kwargs["dtype"]
    elif config.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
        del model_kwargs["dtype"]

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_layer_range(model, config: ExtractionConfig) -> tuple[int, int]:
    """Determine layer range for extraction (defaults to middle third)."""
    if hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers
    elif hasattr(model.config, "n_layer"):
        num_layers = model.config.n_layer
    else:
        raise ValueError("Could not determine number of layers from model config")

    if config.layer_start is not None and config.layer_end is not None:
        return config.layer_start, config.layer_end

    third = num_layers // 3
    layer_start = third
    layer_end = 2 * third

    print(f"Model has {num_layers} layers, extracting from layers {layer_start}-{layer_end}")

    return layer_start, layer_end


def load_contrastive_pairs(config: ExtractionConfig) -> list[dict]:
    """Load contrastive pairs from JSON file."""
    with open(config.data_path, encoding="utf-8") as f:
        data = json.load(f)

    pairs = data.get("pairs", data)
    print(f"Loaded {len(pairs)} contrastive pairs")

    return pairs


def format_training_data(pairs: list[dict]) -> list[tuple[str, str]]:
    """Format contrastive pairs for steering vector training."""
    return [(pair.get("positive", ""), pair.get("negative", "")) for pair in pairs]


def extract_steering_vector(
    model,
    tokenizer,
    training_data: list[tuple[str, str]],
    config: ExtractionConfig,
) -> SteeringVector:
    """Extract steering vector using the steering-vectors library."""
    layer_start, layer_end = get_layer_range(model, config)

    print(f"Training steering vector on {len(training_data)} pairs...")
    print(f"Layers: {layer_start} to {layer_end}")

    steering_vector = train_steering_vector(
        model=model,
        tokenizer=tokenizer,
        training_samples=training_data,
        layers=list(range(layer_start, layer_end + 1)),
        show_progress=True,
    )

    return steering_vector


def save_steering_vector(
    steering_vector: SteeringVector,
    config: ExtractionConfig,
    metadata: dict,
):
    """Save steering vector and metadata."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    model_name_clean = config.model_name.replace("/", "_").replace("-", "_")
    output_path = config.output_dir / f"{model_name_clean}_sentiment.pt"

    save_data = {
        "layer_activations": steering_vector.layer_activations,
        "layer_type": steering_vector.layer_type,
    }
    torch.save(save_data, output_path)
    print(f"Saved steering vector to: {output_path}")

    metadata_path = config.output_dir / f"{model_name_clean}_sentiment_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")

    return output_path


def main():
    """CLI entrypoint for steering vector extraction."""
    parser = argparse.ArgumentParser(
        description="Extract steering vectors from contrastive pairs"
    )

    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to contrastive pairs JSON file"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory to save steering vectors"
    )
    parser.add_argument(
        "--layer-start", type=int, default=None,
        help="Start layer for extraction"
    )
    parser.add_argument(
        "--layer-end", type=int, default=None,
        help="End layer for extraction"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--load-in-8bit", action="store_true",
        help="Load model in 8-bit precision"
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true",
        help="Load model in 4-bit precision"
    )

    args = parser.parse_args()

    config = ExtractionConfig(
        model_name=args.model,
        data_path=args.data,
        output_dir=Path(args.output_dir),
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        torch_dtype=args.dtype,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    model, tokenizer = load_model_and_tokenizer(config)
    layer_start, layer_end = get_layer_range(model, config)

    pairs = load_contrastive_pairs(config)
    training_data = format_training_data(pairs)

    steering_vector = extract_steering_vector(
        model=model,
        tokenizer=tokenizer,
        training_data=training_data,
        config=config,
    )

    metadata = {
        "model_name": config.model_name,
        "concept": "sentiment",
        "direction": "positive",
        "num_pairs": len(pairs),
        "layer_start": layer_start,
        "layer_end": layer_end,
        "data_path": str(config.data_path),
        "dtype": config.torch_dtype,
    }

    output_path = save_steering_vector(steering_vector, config, metadata)

    print("\nExtraction complete!")
    print(f"  Model: {config.model_name}")
    print(f"  Layers: {layer_start}-{layer_end}")
    print(f"  Training pairs: {len(pairs)}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
