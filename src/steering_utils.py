"""
Utilities for loading and applying steering vectors.

This module provides helper functions for working with trained steering
vectors, including generation with steering applied at various strengths.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
from steering_vectors import SteeringVector


@dataclass
class LabeledSteeringVector:
    """A steering vector with an associated label."""

    vector: SteeringVector
    label: str

    def apply(self, model, **kwargs):
        """Delegate to underlying vector's apply method."""
        return self.vector.apply(model, **kwargs)


def load_steering_vector(
    path: Union[str, Path],
    label: Optional[str] = None,
) -> LabeledSteeringVector:
    """
    Load a steering vector from disk.

    Args:
        path: Path to the .pt file
        label: Optional label override (otherwise loaded from metadata or filename)

    Returns:
        LabeledSteeringVector with the vector and its label
    """
    path = Path(path)
    data = torch.load(path, weights_only=False)

    vector = SteeringVector(
        layer_activations=data["layer_activations"],
        layer_type=data.get("layer_type", "decoder_block"),
    )

    # Determine label: explicit > saved in file > metadata > filename
    if label is None:
        label = data.get("label")
    if label is None:
        metadata = load_steering_metadata(path)
        label = metadata.get("concept", metadata.get("label"))
    if label is None:
        # Extract from filename (e.g., "model_sentiment.pt" -> "sentiment")
        label = path.stem.split("_")[-1]

    return LabeledSteeringVector(vector=vector, label=label)


def load_steering_metadata(path: Union[str, Path]) -> dict:
    """Load metadata for a steering vector."""
    path = Path(path)
    metadata_path = path.parent / (path.stem + "_metadata.json")

    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    steering_vector: Optional[Union[SteeringVector, LabeledSteeringVector]] = None,
    steering_strength: float = 1.0,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> str:
    """
    Generate text with optional steering vector applied.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        steering_vector: Optional steering vector to apply
        steering_strength: Multiplier for steering effect (negative inverts direction)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to sample or use greedy decoding

    Returns:
        Generated text (excluding prompt)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }

    # Handle both SteeringVector and LabeledSteeringVector
    sv = steering_vector
    if isinstance(sv, LabeledSteeringVector):
        sv = sv.vector

    if sv is not None and steering_strength != 0:
        with sv.apply(model, multiplier=steering_strength):
            outputs = model.generate(**inputs, **gen_kwargs)
    else:
        outputs = model.generate(**inputs, **gen_kwargs)

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    return generated


def generate_comparison(
    model,
    tokenizer,
    prompt: str,
    steering_vector: Union[SteeringVector, LabeledSteeringVector],
    strengths: Optional[list[float]] = None,
    **gen_kwargs,
) -> pd.DataFrame:
    """
    Generate text at multiple steering strengths for comparison.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        steering_vector: Steering vector to apply
        strengths: List of steering strengths to test (default: [-1, 0, 1])
        **gen_kwargs: Additional generation kwargs

    Returns:
        DataFrame with columns: label, strength, prompt, generation
    """
    if strengths is None:
        strengths = [-1.0, 0.0, 1.0]

    # Get label
    label = (
        steering_vector.label
        if isinstance(steering_vector, LabeledSteeringVector)
        else "unlabeled"
    )

    rows = []
    for strength in strengths:
        generated = generate_with_steering(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            steering_vector=steering_vector,
            steering_strength=strength,
            **gen_kwargs,
        )
        rows.append(
            {
                "label": label,
                "strength": strength,
                "prompt": prompt,
                "generation": generated,
            }
        )

    return pd.DataFrame(rows)


def generate_sweep(
    model,
    tokenizer,
    prompts: list[str],
    steering_vector: Union[SteeringVector, LabeledSteeringVector],
    strengths: Optional[list[float]] = None,
    **gen_kwargs,
) -> pd.DataFrame:
    """
    Generate text for multiple prompts across steering strengths.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of prompts to test
        steering_vector: Steering vector to apply
        strengths: List of steering strengths (default: [-1, 0, 1])
        **gen_kwargs: Additional generation kwargs

    Returns:
        DataFrame with columns: label, strength, prompt, generation
    """
    dfs = []
    for prompt in prompts:
        df = generate_comparison(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            steering_vector=steering_vector,
            strengths=strengths,
            **gen_kwargs,
        )
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def compute_perplexity(
    model,
    tokenizer,
    text: str,
    steering_vector: Optional[Union[SteeringVector, LabeledSteeringVector]] = None,
    steering_strength: float = 0.0,
) -> float:
    """
    Compute perplexity of text under the model (optionally with steering).

    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Text to evaluate
        steering_vector: Optional steering vector
        steering_strength: Steering multiplier

    Returns:
        Perplexity score
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    sv = steering_vector
    if isinstance(sv, LabeledSteeringVector):
        sv = sv.vector

    with torch.no_grad():
        if sv is not None and steering_strength != 0:
            with sv.apply(model, multiplier=steering_strength):
                outputs = model(**inputs, labels=inputs["input_ids"])
        else:
            outputs = model(**inputs, labels=inputs["input_ids"])

    return torch.exp(outputs.loss).item()


def compute_perplexity_sweep(
    model,
    tokenizer,
    texts: list[str],
    steering_vector: Union[SteeringVector, LabeledSteeringVector],
    strengths: Optional[list[float]] = None,
) -> pd.DataFrame:
    """
    Compute perplexity for multiple texts across steering strengths.

    Args:
        model: The language model
        tokenizer: The tokenizer
        texts: List of texts to evaluate
        steering_vector: Steering vector to apply
        strengths: List of steering strengths (default: [-1, 0, 1])

    Returns:
        DataFrame with columns: label, strength, text, perplexity
    """
    if strengths is None:
        strengths = [-1.0, 0.0, 1.0]

    label = (
        steering_vector.label
        if isinstance(steering_vector, LabeledSteeringVector)
        else "unlabeled"
    )

    rows = []
    for text in texts:
        for strength in strengths:
            ppl = compute_perplexity(
                model=model,
                tokenizer=tokenizer,
                text=text,
                steering_vector=steering_vector,
                steering_strength=strength,
            )
            rows.append(
                {
                    "label": label,
                    "strength": strength,
                    "text": text,
                    "perplexity": ppl,
                }
            )

    return pd.DataFrame(rows)
