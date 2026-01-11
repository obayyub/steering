#!/usr/bin/env python3
"""
Create power-seeking-inclination dataset from Anthropic evals.

Format matches the corrigible dataset:
- No "Question:" prefix
- positive = non-power-seeking behavior
- negative = power-seeking behavior
- Balanced A/B distribution (no position bias)
"""

import json
import random
import re
from pathlib import Path


def clean_text(text: str) -> str:
    """Remove any metadata artifacts from text."""
    # Remove META tags like 
