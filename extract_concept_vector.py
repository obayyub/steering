"""
Concept-agnostic steering vector extraction with layer selection.

Works with the generalized contrastive pairs format:
{
    "prompt": "...",
    "positive": "full completion for target behavior",
    "negative": "full completion for opposite behavior",
    "positive_label": "...",
    "negative_label": "..."
}

Supports:
- Single layer extraction (--layer N)
- Layer range (--layer-start N --layer-end M)
- All layers sweep (--all-layers) for layer-wise analysis

Output structure:
results/steering_vectors/<concept>/<model_name>/
├── <concept>_layer_5.pt
├── <concept>_layers_10_20.pt
└── metadata/
    └── <concept>_layer_5.json
"""

import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import train_steering_vector, SteeringVector

from src.steering_utils import format_chat_prompt


@dataclass
class ExtractionConfig:
    model_name: str
    data_path: Path
    output_base: Path
    concept: str

    layer: Optional[int] = None
    layer_start: Optional[int] = None
    layer_end: Optional[int] = None
    all_layers: bool = False

    torch_dtype: str = "bfloat16"
    device: str = "cuda"
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    def __post_init__(self):
        self.data_path = Path(self.data_path)
        self.output_base = Path(self.output_base)

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.torch_dtype_resolved = dtype_map.get(self.torch_dtype, torch.bfloat16)

    @property
    def model_name_clean(self) -> str:
        return self.model_name.replace("/", "_").replace("-", "_")

    @property
    def output_dir(self) -> Path:
        return self.output_base / self.concept / self.model_name_clean

    @property
    def metadata_dir(self) -> Path:
        return self.output_dir / "metadata"


def load_model_and_tokenizer(config: ExtractionConfig):
    print(f"Loading model: {config.model_name}")

    model_kwargs = {
        "torch_dtype": config.torch_dtype_resolved,
        "device_map": config.device,
        "trust_remote_code": True,
    }

    if config.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
        del model_kwargs["torch_dtype"]
    elif config.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
        del model_kwargs["torch_dtype"]

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_num_layers(model) -> int:
    if hasattr(model.config, "num_hidden_layers"):
        return model.config.num_hidden_layers
    elif hasattr(model.config, "n_layer"):
        return model.config.n_layer
    else:
        raise ValueError("Could not determine number of layers from model config")


def get_layer_configs(model, config: ExtractionConfig) -> list[tuple[int, int, str]]:
    """
    Returns list of (layer_start, layer_end, suffix) tuples to extract.

    Suffix is used for output file naming:
    - Single layer: "_layer_5"
    - Range: "_layers_5_10"
    - All layers: "_layer_0", "_layer_1", etc.
    """
    num_layers = get_num_layers(model)

    if config.all_layers:
        print(f"Model has {num_layers} layers, extracting each layer separately")
        return [(i, i, f"_layer_{i}") for i in range(num_layers)]

    if config.layer is not None:
        if config.layer < 0 or config.layer >= num_layers:
            raise ValueError(f"Layer {config.layer} out of range [0, {num_layers})")
        print(f"Extracting from single layer {config.layer}")
        return [(config.layer, config.layer, f"_layer_{config.layer}")]

    if config.layer_start is not None and config.layer_end is not None:
        if config.layer_start < 0 or config.layer_end >= num_layers:
            raise ValueError(f"Layer range [{config.layer_start}, {config.layer_end}] out of bounds")
        print(f"Extracting from layers {config.layer_start} to {config.layer_end}")
        return [(config.layer_start, config.layer_end, f"_layers_{config.layer_start}_{config.layer_end}")]

    # Default: middle third
    third = num_layers // 3
    layer_start = third
    layer_end = 2 * third
    print(f"Model has {num_layers} layers, defaulting to middle third: {layer_start}-{layer_end}")
    return [(layer_start, layer_end, f"_layers_{layer_start}_{layer_end}")]


def load_contrastive_pairs(config: ExtractionConfig) -> list[dict]:
    with open(config.data_path, encoding="utf-8") as f:
        data = json.load(f)

    pairs = data.get("pairs", data) if isinstance(data, dict) else data
    print(f"Loaded {len(pairs)} contrastive pairs")

    if pairs and "positive_label" in pairs[0]:
        print(f"  Positive label: {pairs[0]['positive_label']}")
        print(f"  Negative label: {pairs[0]['negative_label']}")

    return pairs


def format_training_data(pairs: list[dict], tokenizer) -> list[tuple[str, str]]:
    """
    Format contrastive pairs with chat template for extraction.

    The prompt (without answer) goes in user message, then we append
    the answer after the generation prompt so it appears as the
    assistant's response. This matches how the model is trained.
    """
    formatted = []
    for pair in pairs:
        # Extract the prompt (everything before the answer)
        prompt = pair.get("prompt", "")
        if not prompt:
            prompt = pair["positive"].rsplit("(", 1)[0].rstrip()

        # Format with chat template + generation prompt
        messages = [{"role": "user", "content": prompt}]
        base = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        # Append just the answer choices
        pos_answer = pair.get("matching_answer", "(A)")
        neg_answer = pair.get("not_matching_answer", "(B)")

        if "positive" in pair and pair["positive"].endswith(")"):
            pos_answer = "(" + pair["positive"].rsplit("(", 1)[-1]
        if "negative" in pair and pair["negative"].endswith(")"):
            neg_answer = "(" + pair["negative"].rsplit("(", 1)[-1]

        formatted.append((base + pos_answer, base + neg_answer))
    return formatted


def extract_steering_vector(
    model,
    tokenizer,
    training_data: list[tuple[str, str]],
    layer_start: int,
    layer_end: int,
) -> SteeringVector:
    print(f"Training steering vector on {len(training_data)} pairs, layers {layer_start}-{layer_end}...")

    steering_vector = train_steering_vector(
        model=model,
        tokenizer=tokenizer,
        training_samples=training_data,
        layers=list(range(layer_start, layer_end + 1)),
        read_token_index=-2,  # Extract from (A/(B token - the differentiating token
        show_progress=True,
    )

    return steering_vector


def save_steering_vector(
    steering_vector: SteeringVector,
    config: ExtractionConfig,
    metadata: dict,
    suffix: str,
) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.metadata_dir.mkdir(parents=True, exist_ok=True)

    # Save vector
    vector_path = config.output_dir / f"{config.concept}{suffix}.pt"
    save_data = {
        "layer_activations": steering_vector.layer_activations,
        "layer_type": steering_vector.layer_type,
    }
    torch.save(save_data, vector_path)
    print(f"Saved steering vector to: {vector_path}")

    # Save metadata
    metadata_path = config.metadata_dir / f"{config.concept}{suffix}.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return vector_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract steering vectors for any concept from contrastive pairs"
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
        "--concept", type=str, required=True,
        help="Name of the concept being extracted (e.g., sycophancy, formality)"
    )
    parser.add_argument(
        "--output-base", type=str, default="results/steering_vectors",
        help="Base directory for steering vectors (default: results/steering_vectors)"
    )

    # Layer selection (mutually exclusive)
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument(
        "--layer", type=int, default=None,
        help="Extract from a single layer"
    )
    layer_group.add_argument(
        "--all-layers", action="store_true",
        help="Extract each layer separately for layer-wise analysis"
    )

    parser.add_argument(
        "--layer-start", type=int, default=None,
        help="Start layer for range extraction (use with --layer-end)"
    )
    parser.add_argument(
        "--layer-end", type=int, default=None,
        help="End layer for range extraction (use with --layer-start)"
    )

    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
    )
    parser.add_argument(
        "--load-in-8bit", action="store_true",
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true",
    )

    args = parser.parse_args()

    # Validate layer range args
    if (args.layer_start is None) != (args.layer_end is None):
        parser.error("--layer-start and --layer-end must be used together")

    config = ExtractionConfig(
        model_name=args.model,
        data_path=args.data,
        output_base=Path(args.output_base),
        concept=args.concept,
        layer=args.layer,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        all_layers=args.all_layers,
        torch_dtype=args.dtype,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    model, tokenizer = load_model_and_tokenizer(config)
    layer_configs = get_layer_configs(model, config)

    pairs = load_contrastive_pairs(config)
    training_data = format_training_data(pairs, tokenizer)

    output_paths = []
    for layer_start, layer_end, suffix in layer_configs:
        steering_vector = extract_steering_vector(
            model=model,
            tokenizer=tokenizer,
            training_data=training_data,
            layer_start=layer_start,
            layer_end=layer_end,
        )

        metadata = {
            "model_name": config.model_name,
            "concept": config.concept,
            "direction": "positive",
            "num_pairs": len(pairs),
            "layer_start": layer_start,
            "layer_end": layer_end,
            "data_path": str(config.data_path),
            "dtype": config.torch_dtype,
        }

        if pairs and "positive_label" in pairs[0]:
            metadata["positive_label"] = pairs[0]["positive_label"]
            metadata["negative_label"] = pairs[0]["negative_label"]

        output_path = save_steering_vector(steering_vector, config, metadata, suffix)
        output_paths.append(output_path)

    print("\nExtraction complete!")
    print(f"  Model: {config.model_name}")
    print(f"  Concept: {config.concept}")
    print(f"  Training pairs: {len(pairs)}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Vectors extracted: {len(output_paths)}")


if __name__ == "__main__":
    main()
