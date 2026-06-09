from __future__ import annotations

import urllib.error
import urllib.request

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from shared.constants import NUMERIC_COLS, ZERO_INVALID_COLS

# ---------------------------------------------------------------------------
# Task 2.1 — Input Filter
# ---------------------------------------------------------------------------


def input_filter(dataset_url: str) -> pd.DataFrame:
    """Load a crop recommendation CSV from a URL or local file path.

    Preconditions:
        - ``dataset_url`` is a non-empty string representing either an
          HTTP/HTTPS URL or a valid local filesystem path to a CSV file.
        - The CSV must contain at minimum the columns:
          nitrogen, phosphorus, potassium, temperature, humidity, ph,
          rainfall, label.
        - If a remote URL is supplied, network access must be available.

    Postconditions:
        - Returns a :class:`pandas.DataFrame` with at least 100 rows.
        - The DataFrame retains all columns present in the source CSV
          without modification.
        - Raises :class:`urllib.error.URLError` (before any rows are
          read) when the URL cannot be reached.
        - Raises :class:`ValueError` when the loaded dataset contains
          fewer than 100 rows.

    Args:
        dataset_url: HTTP/HTTPS URL or local path to the CSV dataset.

    Returns:
        Raw DataFrame loaded directly from the CSV source.

    Raises:
        urllib.error.URLError: When the remote URL is unreachable.
        ValueError: When the loaded dataset has fewer than 100 rows.
    """
    # Validate connectivity for HTTP/HTTPS URLs before delegating to pandas
    # so that callers receive a URLError (not a generic pandas/urllib3 error).
    if dataset_url.startswith(("http://", "https://")):
        try:
            urllib.request.urlopen(dataset_url, timeout=10)  # noqa: S310
        except urllib.error.URLError as exc:
            raise urllib.error.URLError(
                f"Input Filter: dataset URL unreachable — {dataset_url!r}. "
                f"Original error: {exc.reason}"
            ) from exc

    df: pd.DataFrame = pd.read_csv(dataset_url)

    if len(df) < 100:
        raise ValueError(
            f"Input Filter: dataset has only {len(df)} rows; "
            "at least 100 rows are required for reliable training."
        )

    print(f"[INPUT FILTER] Loaded dataset: shape={df.shape}")
    print(f"[INPUT FILTER] First 5 rows:\n{df.head()}")

    return df


# ---------------------------------------------------------------------------
# Task 2.3 — Cleaning Filter
# ---------------------------------------------------------------------------


def cleaning_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw crop DataFrame by removing duplicates, imputing missing
    values, and clipping outliers.

    Preconditions:
        - ``df`` is a non-empty DataFrame produced by :func:`input_filter`.
        - ``df`` contains all columns listed in ``NUMERIC_COLS``:
          nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall.
        - ``df`` has at least 100 rows.

    Postconditions:
        - No duplicate rows remain in the returned DataFrame.
        - No NaN values remain in any column listed in ``NUMERIC_COLS``.
        - Each numeric column has been IQR-clipped to
          [Q1 - 1.5 x IQR, Q3 + 1.5 x IQR].
        - Row count may decrease (duplicates removed) but column count is
          unchanged.
        - Non-numeric columns (e.g. ``label``) are preserved without
          modification.

    Args:
        df: Raw DataFrame from :func:`input_filter`.

    Returns:
        Cleaned DataFrame ready for feature engineering.

    Raises:
        AssertionError: When NaN values remain in numeric columns after
            imputation (indicates a logic error).
    """
    original_len = len(df)

    # Step 1: Drop duplicate rows
    df = df.drop_duplicates()
    dropped_dupes = original_len - len(df)
    print(f"[CLEANING FILTER] Dropped {dropped_dupes} duplicate rows.")

    # Step 2: Replace invalid zeros with NaN where 0 is physically impossible
    for col in ZERO_INVALID_COLS:
        if col in df.columns:
            zeros_replaced = int((df[col] == 0).sum())
            df[col] = df[col].replace(0, np.nan)
            if zeros_replaced:
                print(
                    f"[CLEANING FILTER] Replaced {zeros_replaced} zero(s) "
                    f"with NaN in column '{col}'."
                )

    # Step 3: Median imputation on numeric columns
    imputer = SimpleImputer(strategy="median")
    df[NUMERIC_COLS] = imputer.fit_transform(df[NUMERIC_COLS])

    # Step 4: IQR-based outlier clipping.
    # Loop invariant: each column is processed independently;
    # previously processed columns remain unchanged.
    for col in NUMERIC_COLS:
        q1: float = df[col].quantile(0.25)
        q3: float = df[col].quantile(0.75)
        iqr: float = q3 - q1
        lower: float = q1 - 1.5 * iqr
        upper: float = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower, upper)

    # Postcondition guard
    assert not df[NUMERIC_COLS].isna().any().any(), (
        "Cleaning Filter postcondition violated: NaN values remain in numeric columns."
    )

    print(
        f"[CLEANING FILTER] Done. Output shape={df.shape}. "
        "No NaN in numeric columns: confirmed."
    )
    return df


# ---------------------------------------------------------------------------
# Task 2.5 — Feature Engineering Filter
# ---------------------------------------------------------------------------


def feature_engineering_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Derive new features, one-hot encode the crop label, and prepare
    ``disease_label`` and ``yield_kg_per_ha`` target columns.

    Preconditions:
        - ``df`` has passed :func:`cleaning_filter` (no NaNs in numeric cols).
        - ``df`` contains a ``label`` column with string crop-type values.
        - All ``NUMERIC_COLS`` are present and free of NaN.

    Postconditions:
        - ``npk_ratio`` column is added:
          ``(nitrogen + phosphorus) / (potassium + 1e-6)``.
        - ``temp_humidity_index`` column is added:
          ``temperature x humidity / 100.0``.
        - ``disease_label`` column is added as a copy of the original
          ``label`` column (used as proxy target for the classifier).
        - ``yield_kg_per_ha`` column is added as a synthetic regression
          target: ``2000 + N*10 + P*5 + rainfall*2`` plus small Gaussian
          noise (random_state=42).
        - ``label`` is one-hot encoded into ``crop_type_*`` binary columns
          and the original ``label`` column is dropped.
        - No NaN values are introduced by any of the above operations.

    Args:
        df: Cleaned DataFrame from :func:`cleaning_filter`.

    Returns:
        Feature-engineered DataFrame ready for scaling.

    Raises:
        AssertionError: When NaN values are detected in the output.
    """
    df = df.copy()

    # Preserve original label as disease proxy BEFORE one-hot encoding
    df["disease_label"] = df["label"]

    # Synthetic yield target (no real yield column in the Crop Recommendation
    # dataset — generated deterministically from soil/climate features).
    rng = np.random.default_rng(42)
    df["yield_kg_per_ha"] = (
        2000.0
        + df["nitrogen"] * 10.0
        + df["phosphorus"] * 5.0
        + df["rainfall"] * 2.0
        + rng.normal(loc=0.0, scale=50.0, size=len(df))
    )

    # Derived features
    df["npk_ratio"] = (df["nitrogen"] + df["phosphorus"]) / (
        df["potassium"] + 1e-6
    )
    df["temp_humidity_index"] = df["temperature"] * df["humidity"] / 100.0

    # One-hot encode the original 'label' column into crop_type_* columns
    df = pd.get_dummies(df, columns=["label"], prefix="crop_type", drop_first=False)

    # Postcondition guard: no NaN introduced
    assert not df.isna().any().any(), (
        "Feature Engineering Filter postcondition violated: "
        "NaN values detected in the output DataFrame."
    )

    n_crop_cols = sum(1 for c in df.columns if c.startswith("crop_type_"))
    print(
        f"[FEATURE ENGINEERING FILTER] Added npk_ratio, temp_humidity_index, "
        f"disease_label, yield_kg_per_ha. "
        f"One-hot encoded 'label' into {n_crop_cols} crop_type_* columns. "
        f"Output shape={df.shape}."
    )
    return df


# ---------------------------------------------------------------------------
# Task 2.7 — Scaling Filter
# ---------------------------------------------------------------------------


def scaling_filter(
    df: pd.DataFrame,
) -> tuple[np.ndarray, pd.Series, pd.Series, StandardScaler]:
    """Separate targets from the feature matrix and apply StandardScaler.

    Preconditions:
        - ``df`` has passed :func:`feature_engineering_filter`.
        - Columns ``disease_label`` (str Series) and ``yield_kg_per_ha``
          (float Series) are present in ``df``.
        - All remaining columns after dropping targets are numeric and
          free of NaN.

    Postconditions:
        - Returns a 4-tuple ``(X_scaled, y_disease, y_yield, scaler)``.
        - ``X_scaled`` is a 2-D ``float64`` NumPy array with shape
          ``(n_samples, n_features)`` containing no NaN or Inf values.
        - ``y_disease`` is a :class:`pandas.Series` of string crop/disease
          labels aligned row-for-row with ``X_scaled``.
        - ``y_yield`` is a :class:`pandas.Series` of positive float yield
          values aligned row-for-row with ``X_scaled``.
        - ``scaler`` is the :class:`~sklearn.preprocessing.StandardScaler`
          instance fitted on the feature matrix; it must be serialised
          alongside the trained models for inference-time consistency.

    Args:
        df: Feature-engineered DataFrame from
            :func:`feature_engineering_filter`.

    Returns:
        Tuple of ``(X_scaled, y_disease, y_yield, scaler)``.

    Raises:
        AssertionError: When ``X_scaled`` contains NaN or Inf values after
            scaling (indicates a preprocessing logic error).
    """
    df = df.copy()

    # Extract target columns
    y_disease: pd.Series = df["disease_label"].reset_index(drop=True)
    y_yield: pd.Series = df["yield_kg_per_ha"].reset_index(drop=True)

    # Drop both target columns from the feature matrix
    feature_df = df.drop(columns=["disease_label", "yield_kg_per_ha"])

    # Convert all remaining columns to float64.
    # pd.get_dummies may produce bool dtype on older pandas versions;
    # StandardScaler requires numeric dtype.
    feature_df = feature_df.astype(float)

    # Fit StandardScaler and transform
    scaler = StandardScaler()
    X_scaled: np.ndarray = scaler.fit_transform(feature_df)

    # Postcondition guards
    assert not np.isnan(X_scaled).any(), (
        "Scaling Filter postcondition violated: NaN values in X_scaled."
    )
    assert not np.isinf(X_scaled).any(), (
        "Scaling Filter postcondition violated: Inf values in X_scaled."
    )

    print(
        f"[SCALING FILTER] X_scaled shape={X_scaled.shape}. "
        f"y_disease classes={y_disease.nunique()}. "
        f"y_yield range=[{y_yield.min():.1f}, {y_yield.max():.1f}]. "
        "No NaN/Inf in X_scaled: confirmed."
    )
    return X_scaled, y_disease, y_yield, scaler


# ---------------------------------------------------------------------------
# Task 2.8 — Training Filter (Disease Classifier)
# ---------------------------------------------------------------------------


def training_filter_disease(
    X: np.ndarray,
    y: pd.Series,
) -> "RandomForestClassifier":
    """Train a RandomForest classifier for crop disease detection.

    Preconditions:
        - ``X`` is a 2-D ``float64`` NumPy array with no NaN or Inf values,
          shape ``(n, m)`` where ``n >= 100``.
        - ``y`` is a 1-D string :class:`pandas.Series` of disease labels,
          ``len(y) == n``.
        - At least 2 unique classes are present in ``y``.

    Postconditions:
        - Returns a fitted :class:`~sklearn.ensemble.RandomForestClassifier`
          with a ``classes_`` attribute.
        - Test accuracy >= 0.70 (quality gate); raises :class:`ValueError`
          if the gate is not met.
        - Prints accuracy and F1-macro scores to stdout.

    Args:
        X: Scaled feature matrix from :func:`scaling_filter`.
        y: Series of disease label strings aligned row-for-row with ``X``.

    Returns:
        Fitted ``RandomForestClassifier`` instance.

    Raises:
        ValueError: When test accuracy falls below the 0.70 quality gate.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc: float = accuracy_score(y_test, y_pred)
    f1: float = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"[TRAINING FILTER - DISEASE] Test Accuracy: {acc:.4f}")
    print(f"[TRAINING FILTER - DISEASE] F1-Macro: {f1:.4f}")

    if acc < 0.70:
        raise ValueError(
            f"Disease model accuracy {acc:.4f} below quality gate 0.70"
        )

    return model


# ---------------------------------------------------------------------------
# Task 2.10 — Training Filter (Yield Regressor)
# ---------------------------------------------------------------------------


def training_filter_yield(
    X: np.ndarray,
    y: pd.Series,
) -> "GradientBoostingRegressor":
    """Train a GradientBoosting regressor for crop yield prediction.

    Preconditions:
        - ``X`` is a 2-D ``float64`` NumPy array with no NaN or Inf values.
        - ``y`` is a 1-D :class:`pandas.Series` of positive float values
          (kg/ha).
        - ``len(X) == len(y) >= 100``.

    Postconditions:
        - Returns a fitted :class:`~sklearn.ensemble.GradientBoostingRegressor`.
        - Test RMSE <= 500 kg/ha (quality gate); raises :class:`ValueError`
          if the gate is not met.
        - Prints RMSE to stdout.

    Args:
        X: Scaled feature matrix from :func:`scaling_filter`.
        y: Series of positive yield values (kg/ha) aligned with ``X``.

    Returns:
        Fitted ``GradientBoostingRegressor`` instance.

    Raises:
        ValueError: When test RMSE exceeds the 500 kg/ha quality gate.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )
    model.fit(X_train, y_train)

    rmse: float = float(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))
    print(f"[TRAINING FILTER - YIELD] Test RMSE: {rmse:.2f} kg/ha")

    if rmse > 500:
        raise ValueError(
            f"Yield model RMSE {rmse:.2f} above quality gate 500 kg/ha"
        )

    return model


# ---------------------------------------------------------------------------
# Task 2.12 — Output Filter (Model Serialisation)
# ---------------------------------------------------------------------------


def output_filter(ctx: "PipelineContext", output_dir: str) -> dict:
    """Serialise trained models and scaler to disk; return evaluation metrics.

    Preconditions:
        - ``ctx.disease_model`` and ``ctx.yield_model`` are fitted sklearn
          estimators.
        - ``ctx.scaler`` is a fitted :class:`~sklearn.preprocessing.StandardScaler`.
        - ``output_dir`` is a writable directory path (created if absent).

    Postconditions:
        - ``disease_model.pkl``, ``yield_model.pkl``, and ``scaler.pkl`` are
          written to ``output_dir`` via :func:`joblib.dump`.
        - All three files exist on disk with size > 0 bytes.
        - Returns a ``metrics`` dict with keys
          ``disease_accuracy``, ``disease_f1_macro``, ``yield_rmse``
          sourced from ``ctx.metrics``.

    Args:
        ctx: :class:`~pipeline.main_pipeline.PipelineContext` populated by
            the preceding training filters.
        output_dir: Path to the directory where ``.pkl`` files will be saved.

    Returns:
        ``dict`` with keys ``disease_accuracy``, ``disease_f1_macro``,
        ``yield_rmse``.

    Raises:
        AssertionError: When any serialised file is missing or has zero size.
    """
    import os

    import joblib
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
    from sklearn.model_selection import train_test_split

    os.makedirs(output_dir, exist_ok=True)

    # --- Compute metrics from the disease model ---
    X_d_train, X_d_test, y_d_train, y_d_test = train_test_split(
        ctx.X_scaled, ctx.y_disease, test_size=0.2, random_state=42, stratify=ctx.y_disease
    )
    y_d_pred = ctx.disease_model.predict(X_d_test)
    disease_accuracy: float = float(accuracy_score(y_d_test, y_d_pred))
    disease_f1_macro: float = float(
        f1_score(y_d_test, y_d_pred, average="macro", zero_division=0)
    )

    # --- Compute metrics from the yield model ---
    X_y_train, X_y_test, y_y_train, y_y_test = train_test_split(
        ctx.X_scaled, ctx.y_yield, test_size=0.2, random_state=42
    )
    yield_rmse: float = float(
        np.sqrt(mean_squared_error(y_y_test, ctx.yield_model.predict(X_y_test)))
    )

    # --- Serialise all three artefacts ---
    disease_path = os.path.join(output_dir, "disease_model.pkl")
    yield_path = os.path.join(output_dir, "yield_model.pkl")
    scaler_path = os.path.join(output_dir, "scaler.pkl")

    joblib.dump(ctx.disease_model, disease_path)
    joblib.dump(ctx.yield_model, yield_path)
    joblib.dump(ctx.scaler, scaler_path)

    # --- Postcondition guards ---
    for path in (disease_path, yield_path, scaler_path):
        assert os.path.isfile(path), (
            f"Output Filter postcondition violated: expected file not found: {path}"
        )
        assert os.path.getsize(path) > 0, (
            f"Output Filter postcondition violated: file is empty: {path}"
        )

    metrics: dict = {
        "disease_accuracy": disease_accuracy,
        "disease_f1_macro": disease_f1_macro,
        "yield_rmse": yield_rmse,
    }

    # Cache metrics on context for downstream use
    ctx.metrics = metrics

    print(
        f"[OUTPUT FILTER] Models saved to '{output_dir}'. "
        f"disease_accuracy={disease_accuracy:.4f}, "
        f"disease_f1_macro={disease_f1_macro:.4f}, "
        f"yield_rmse={yield_rmse:.2f} kg/ha."
    )
    return metrics
