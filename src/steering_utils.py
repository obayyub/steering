"""Utilities for loading and applying steering vectors."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
from steering_vectors import SteeringVector
from transformers import pipeline


@dataclass
class LabeledSteeringVector:
    """A steering vector with an associated label."""

    vector: SteeringVector
    label: str

    def apply(self, model, **kwargs):
        return self.vector.apply(model, **kwargs)


def load_steering_vector(
    path: Union[str, Path],
    label: Optional[str] = None,
) -> LabeledSteeringVector:
    """Load a steering vector from disk."""
    path = Path(path)
    data = torch.load(path, weights_only=False)

    vector = SteeringVector(
        layer_activations=data["layer_activations"],
        layer_type=data.get("layer_type", "decoder_block"),
    )

    if label is None:
        label = data.get("label")
    if label is None:
        metadata = load_steering_metadata(path)
        label = metadata.get("concept", metadata.get("label"))
    if label is None:
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


def format_chat_prompt(
    tokenizer,
    prompt: str,
    system_prompt: Optional[str] = None,
    enable_thinking: bool = False,
) -> str:
    """Format a prompt using the tokenizer's chat template."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    steering_vector: Optional[Union[SteeringVector, LabeledSteeringVector]] = None,
    steering_strength: float = 1.0,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    do_sample: bool = True,
    use_chat_template: bool = True,
    enable_thinking: bool = False,
    system_prompt: Optional[str] = None,
) -> str:
    """Generate text with optional steering vector applied."""
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        formatted_prompt = format_chat_prompt(
            tokenizer, prompt, system_prompt, enable_thinking
        )
    else:
        formatted_prompt = prompt

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }

    sv = steering_vector.vector if isinstance(steering_vector, LabeledSteeringVector) else steering_vector

    if sv is not None and steering_strength != 0:
        with sv.apply(model, multiplier=steering_strength):
            outputs = model.generate(**inputs, **gen_kwargs)
    else:
        outputs = model.generate(**inputs, **gen_kwargs)

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


def generate_comparison(
    model,
    tokenizer,
    prompt: str,
    steering_vector: Union[SteeringVector, LabeledSteeringVector],
    strengths: Optional[list[float]] = None,
    **gen_kwargs,
) -> pd.DataFrame:
    """Generate text at multiple steering strengths for comparison."""
    if strengths is None:
        strengths = [-1.0, 0.0, 1.0]

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
        rows.append({
            "label": label,
            "strength": strength,
            "prompt": prompt,
            "generation": generated,
        })

    return pd.DataFrame(rows)


def generate_sweep(
    model,
    tokenizer,
    prompts: list[str],
    steering_vector: Union[SteeringVector, LabeledSteeringVector],
    strengths: Optional[list[float]] = None,
    **gen_kwargs,
) -> pd.DataFrame:
    """Generate text for multiple prompts across steering strengths."""
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


_sentiment_pipeline = None


def get_sentiment_pipeline(device: Optional[str] = None):
    """Get or create the sentiment analysis pipeline."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device,
        )
    return _sentiment_pipeline


def analyze_sentiment(
    texts: list[str],
    device: Optional[str] = None,
) -> list[dict]:
    """Analyze sentiment of texts."""
    pipe = get_sentiment_pipeline(device)
    return pipe(texts, truncation=True, max_length=512)


def analyze_generations(
    df: pd.DataFrame,
    generation_column: str = "generation",
    device: Optional[str] = None,
) -> pd.DataFrame:
    """Add sentiment analysis to a generations DataFrame."""
    texts = df[generation_column].tolist()
    results = analyze_sentiment(texts, device=device)

    df = df.copy()
    df["sentiment_label"] = [r["label"] for r in results]
    df["sentiment_score"] = [r["score"] for r in results]
    df["p_positive"] = [
        r["score"] if r["label"] == "POSITIVE" else 1 - r["score"] for r in results
    ]

    return df
