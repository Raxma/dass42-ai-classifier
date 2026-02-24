# dass_bot/predict.py
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import joblib

# -------------------------------------------------
# Config
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"   # expects depression_best.pkl, anxiety_best.pkl, stress_best.pkl
FEATURE_COLS = [f"Q{i}A" for i in range(1, 43)]  # Q1A..Q42A

# -------------------------------------------------
# Internal helpers
# -------------------------------------------------
def _validate_answers(answers: List[int]) -> None:
    if not isinstance(answers, list):
        raise TypeError("answers must be a list of 42 ints (values 1..4).")
    if len(answers) != 42:
        raise ValueError(f"answers must have length 42 (got {len(answers)}).")
    if not all(v in (1, 2, 3, 4) for v in answers):
        bad = [v for v in answers if v not in (1, 2, 3, 4)]
        raise ValueError(f"All answers must be in [1, 2, 3, 4]. Found invalid values: {bad}")

def _to_frame(answers: List[int]) -> pd.DataFrame:
    """Turn a 42-length list into a single-row DF with Q1A..Q42A columns."""
    return pd.DataFrame([answers], columns=FEATURE_COLS)

@lru_cache(maxsize=None)
def _load_model(model_name: str):
    """
    Lazily load and cache a model by its logical name:
    - 'depression' -> models/depression_best.pkl
    - 'anxiety'    -> models/anxiety_best.pkl
    - 'stress'     -> models/stress_best.pkl
    """
    path = MODEL_DIR / f"{model_name}_best.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}. Did you train & save it?")
    return joblib.load(path)

def _predict_generic(
    model_name: str,
    answers: List[int],
    return_proba: bool = False
) -> Dict[str, Any]:
    """Generic predictor used by the public functions below."""
    _validate_answers(answers)
    X = _to_frame(answers)
    model = _load_model(model_name)

    pred = model.predict(X)[0]
    out: Dict[str, Any] = {"label": pred}

    if return_proba and hasattr(model, "predict_proba"):
        # predict_proba returns an array of shape (n_samples, n_classes)
        probs = model.predict_proba(X)[0]
        classes = list(model.classes_)
        out["proba"] = {cls: float(p) for cls, p in zip(classes, probs)}

    return out

# -------------------------------------------------
# Public API
# -------------------------------------------------
def predict_depression(answers: List[int], return_proba: bool = False) -> Dict[str, Any]:
    """Predict Depression label (and optionally class probabilities) from 42 answers."""
    return _predict_generic("depression", answers, return_proba)

def predict_anxiety(answers: List[int], return_proba: bool = False) -> Dict[str, Any]:
    """Predict Anxiety label (and optionally class probabilities) from 42 answers."""
    return _predict_generic("anxiety", answers, return_proba)

def predict_stress(answers: List[int], return_proba: bool = False) -> Dict[str, Any]:
    """Predict Stress label (and optionally class probabilities) from 42 answers."""
    return _predict_generic("stress", answers, return_proba)

def predict_all(answers: List[int], return_proba: bool = False) -> Dict[str, Any]:
    """
    Convenience function: run all three models and return a single dict.
    Keys: depression, anxiety, stress
    """
    return {
        "depression": predict_depression(answers, return_proba),
        "anxiety": predict_anxiety(answers, return_proba),
        "stress": predict_stress(answers, return_proba),
    }
