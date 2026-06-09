"""Pipeline orchestrator for the Agriculture ML System.

This module wires all Pipe-and-Filter stages together and exposes a single
:func:`run_full_pipeline` entry point for training both the disease classifier
and the yield regressor end-to-end.

Preconditions for :func:`run_full_pipeline`:
    - ``dataset_url`` is a reachable HTTP/HTTPS URL or a valid local path.
    - ``output_dir`` exists or will be created by the function.

Postconditions for :func:`run_full_pipeline`:
    - ``disease_model.pkl``, ``yield_model.pkl``, ``scaler.pkl`` are saved
      to ``output_dir``.
    - Returns a metrics dict with keys:
      ``disease_accuracy``, ``disease_f1_macro``, ``yield_rmse``.
    - Pipeline is deterministic: same URL + ``random_state=42`` → same metrics.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pipeline.filters import (
    cleaning_filter,
    feature_engineering_filter,
    input_filter,
    output_filter,
    scaling_filter,
    training_filter_disease,
    training_filter_yield,
)


@dataclass
class PipelineContext:
    """Mutable state container passed by reference between pipeline filters.

    Each filter reads from earlier fields and writes to its own output
    field(s) without overwriting fields set by previous stages.
    """

    raw_df: pd.DataFrame | None = None
    cleaned_df: pd.DataFrame | None = None
    engineered_df: pd.DataFrame | None = None
    X_scaled: np.ndarray | None = None
    y_disease: pd.Series | None = None
    y_yield: pd.Series | None = None
    scaler: StandardScaler | None = None
    disease_model: object | None = None
    yield_model: object | None = None
    metrics: dict = field(default_factory=dict)


def run_full_pipeline(dataset_url: str, output_dir: str = "./models") -> dict:
    """Execute the complete Pipe-and-Filter ML training pipeline.

    Preconditions:
        - ``dataset_url`` is a reachable HTTP/HTTPS URL or a valid local
          filesystem path to a CSV file.
        - ``output_dir`` exists or can be created (write permission required).

    Postconditions:
        - ``disease_model.pkl``, ``yield_model.pkl``, and ``scaler.pkl`` are
          saved to ``output_dir``.
        - Returns a ``metrics`` dict with the following keys:
          ``disease_accuracy`` (float), ``disease_f1_macro`` (float),
          ``yield_rmse`` (float).
        - Pipeline is deterministic: the same ``dataset_url`` with
          ``random_state=42`` always produces identical metric values.

    Loop Invariants:
        - Each filter receives the output of the previous filter unmodified.
        - ``PipelineContext`` is passed by reference; filters append to it
          without overwriting fields set by earlier stages.

    Args:
        dataset_url: HTTP/HTTPS URL or local path to the crop CSV dataset.
        output_dir: Directory where serialised ``.pkl`` files will be written.
            Created automatically if it does not exist.

    Returns:
        ``dict`` with keys ``disease_accuracy``, ``disease_f1_macro``,
        ``yield_rmse``.

    Raises:
        urllib.error.URLError: When ``dataset_url`` is an unreachable URL.
        ValueError: When the dataset has fewer than 100 rows, or when either
            trained model fails its quality gate.
    """
    os.makedirs(output_dir, exist_ok=True)
    ctx = PipelineContext()

    # PIPE 1: Ingestion
    ctx.raw_df = input_filter(dataset_url)

    # PIPE 2: Cleaning
    ctx.cleaned_df = cleaning_filter(ctx.raw_df)

    # PIPE 3: Feature Engineering
    ctx.engineered_df = feature_engineering_filter(ctx.cleaned_df)

    # PIPE 4: Scaling — produces X_scaled, y_disease, y_yield, scaler
    ctx.X_scaled, ctx.y_disease, ctx.y_yield, ctx.scaler = scaling_filter(
        ctx.engineered_df
    )

    # PIPE 5a: Train Disease Classifier
    ctx.disease_model = training_filter_disease(ctx.X_scaled, ctx.y_disease)

    # PIPE 5b: Train Yield Regressor
    ctx.yield_model = training_filter_yield(ctx.X_scaled, ctx.y_yield)

    # PIPE 6: Serialise models + collect metrics
    ctx.metrics = output_filter(ctx, output_dir)

    return ctx.metrics
