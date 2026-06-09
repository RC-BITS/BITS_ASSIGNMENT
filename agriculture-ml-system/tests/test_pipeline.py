"""Unit tests for all pipeline filters in pipeline/filters.py.

Tests cover:
- input_filter: valid CSV load, ValueError on <100 rows
- cleaning_filter: duplicate removal, zero replacement, no NaN after cleaning
- feature_engineering_filter: npk_ratio, temp_humidity_index, one-hot encoding,
  disease_label, yield_kg_per_ha
- scaling_filter: shape, no NaN/Inf, y_disease and y_yield lengths
- training_filter_disease: fitted RandomForestClassifier, quality gate ValueError
- training_filter_yield: fitted GradientBoostingRegressor, quality gate ValueError
- output_filter: .pkl files created, metrics dict keys correct
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from pipeline.filters import (
    cleaning_filter,
    feature_engineering_filter,
    input_filter,
    output_filter,
    scaling_filter,
    training_filter_disease,
    training_filter_yield,
)
from pipeline.main_pipeline import PipelineContext
from shared.constants import NUMERIC_COLS, ZERO_INVALID_COLS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CROP_TYPES_SUBSET: list[str] = ["rice", "wheat", "maize", "chickpea", "kidneybeans"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """200-row DataFrame mimicking the Crop Recommendation dataset structure.

    Column ranges match the real dataset:
        nitrogen: 0-140, phosphorus: 5-145, potassium: 5-205,
        temperature: 8-44, humidity: 14-100, ph: 3.5-10, rainfall: 20-300,
        label: one of 5 crop types (evenly distributed, 40 rows each).
    """
    rng = np.random.default_rng(0)
    n: int = 200

    data: dict = {
        "nitrogen":    rng.uniform(0, 140, n),
        "phosphorus":  rng.uniform(5, 145, n),
        "potassium":   rng.uniform(5, 205, n),
        "temperature": rng.uniform(8, 44, n),
        "humidity":    rng.uniform(14, 100, n),
        "ph":          rng.uniform(3.5, 10, n),
        "rainfall":    rng.uniform(20, 300, n),
        "label": np.tile(CROP_TYPES_SUBSET, n // len(CROP_TYPES_SUBSET)),
    }
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_csv(synthetic_df: pd.DataFrame, tmp_path) -> str:
    """Write synthetic_df to a temporary CSV and return the file path."""
    csv_path: str = str(tmp_path / "synthetic_crop.csv")
    synthetic_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def mini_pipeline_ctx(synthetic_df: pd.DataFrame) -> PipelineContext:
    """Run the full mini-pipeline on 200 synthetic rows; return a populated context."""
    ctx = PipelineContext()
    ctx.raw_df = synthetic_df.copy()
    ctx.cleaned_df = cleaning_filter(ctx.raw_df)
    ctx.engineered_df = feature_engineering_filter(ctx.cleaned_df)
    ctx.X_scaled, ctx.y_disease, ctx.y_yield, ctx.scaler = scaling_filter(
        ctx.engineered_df
    )
    ctx.disease_model = training_filter_disease(ctx.X_scaled, ctx.y_disease)
    ctx.yield_model = training_filter_yield(ctx.X_scaled, ctx.y_yield)
    return ctx


# ---------------------------------------------------------------------------
# input_filter tests
# ---------------------------------------------------------------------------


class TestInputFilter:
    def test_loads_valid_csv(self, synthetic_csv: str, synthetic_df: pd.DataFrame) -> None:
        """input_filter loads a valid local CSV and returns a DataFrame."""
        df: pd.DataFrame = input_filter(synthetic_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(synthetic_df)

    def test_returns_all_columns(self, synthetic_csv: str, synthetic_df: pd.DataFrame) -> None:
        """input_filter preserves all columns from the source CSV."""
        df: pd.DataFrame = input_filter(synthetic_csv)
        assert set(df.columns) == set(synthetic_df.columns)

    def test_raises_value_error_on_fewer_than_100_rows(self, tmp_path) -> None:
        """input_filter raises ValueError when the CSV has fewer than 100 rows."""
        rng = np.random.default_rng(1)
        small_df = pd.DataFrame({
            "nitrogen": rng.uniform(0, 140, 50),
            "phosphorus": rng.uniform(5, 145, 50),
            "potassium": rng.uniform(5, 205, 50),
            "temperature": rng.uniform(8, 44, 50),
            "humidity": rng.uniform(14, 100, 50),
            "ph": rng.uniform(3.5, 10, 50),
            "rainfall": rng.uniform(20, 300, 50),
            "label": ["rice"] * 50,
        })
        csv_path: str = str(tmp_path / "small.csv")
        small_df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError):
            input_filter(csv_path)

    def test_raises_value_error_exact_99_rows(self, tmp_path) -> None:
        """input_filter raises ValueError for exactly 99 rows."""
        rng = np.random.default_rng(2)
        df_99 = pd.DataFrame({
            "nitrogen": rng.uniform(0, 140, 99),
            "phosphorus": rng.uniform(5, 145, 99),
            "potassium": rng.uniform(5, 205, 99),
            "temperature": rng.uniform(8, 44, 99),
            "humidity": rng.uniform(14, 100, 99),
            "ph": rng.uniform(3.5, 10, 99),
            "rainfall": rng.uniform(20, 300, 99),
            "label": ["wheat"] * 99,
        })
        csv_path: str = str(tmp_path / "df_99.csv")
        df_99.to_csv(csv_path, index=False)
        with pytest.raises(ValueError):
            input_filter(csv_path)


# ---------------------------------------------------------------------------
# cleaning_filter tests
# ---------------------------------------------------------------------------


class TestCleaningFilter:
    def test_drops_duplicates(self, synthetic_df: pd.DataFrame) -> None:
        """cleaning_filter removes duplicate rows."""
        extra = synthetic_df.iloc[:10].copy()
        df_with_dupes = pd.concat([synthetic_df, extra], ignore_index=True)
        assert len(df_with_dupes) == 210

        cleaned: pd.DataFrame = cleaning_filter(df_with_dupes)
        assert len(cleaned) <= 200

    def test_zeros_replaced_in_zero_invalid_cols(self, synthetic_df: pd.DataFrame) -> None:
        """cleaning_filter replaces 0 with NaN then imputes in ZERO_INVALID_COLS."""
        df = synthetic_df.copy()
        for col in ZERO_INVALID_COLS:
            if col in df.columns:
                df.loc[0, col] = 0

        cleaned: pd.DataFrame = cleaning_filter(df)

        # Zeros should be replaced and imputed — no NaN remains
        for col in ZERO_INVALID_COLS:
            if col in df.columns:
                assert not cleaned[col].isna().any(), (
                    f"NaN found in '{col}' after cleaning"
                )

    def test_no_nan_after_cleaning(self, synthetic_df: pd.DataFrame) -> None:
        """cleaning_filter leaves no NaN in NUMERIC_COLS."""
        df = synthetic_df.copy()
        for col in NUMERIC_COLS:
            df.loc[5, col] = np.nan

        cleaned: pd.DataFrame = cleaning_filter(df)
        assert not cleaned[NUMERIC_COLS].isna().any().any()

    def test_column_count_unchanged(self, synthetic_df: pd.DataFrame) -> None:
        """cleaning_filter does not add or remove columns."""
        cleaned: pd.DataFrame = cleaning_filter(synthetic_df.copy())
        assert set(cleaned.columns) == set(synthetic_df.columns)

    def test_label_column_preserved(self, synthetic_df: pd.DataFrame) -> None:
        """cleaning_filter preserves the non-numeric 'label' column."""
        cleaned: pd.DataFrame = cleaning_filter(synthetic_df.copy())
        assert "label" in cleaned.columns


# ---------------------------------------------------------------------------
# feature_engineering_filter tests
# ---------------------------------------------------------------------------


class TestFeatureEngineeringFilter:
    def test_npk_ratio_computed_correctly(self, synthetic_df: pd.DataFrame) -> None:
        """feature_engineering_filter computes npk_ratio = (N+P)/(K+1e-6)."""
        cleaned: pd.DataFrame = cleaning_filter(synthetic_df.copy())
        engineered: pd.DataFrame = feature_engineering_filter(cleaned)

        expected = (cleaned["nitrogen"] + cleaned["phosphorus"]) / (
            cleaned["potassium"] + 1e-6
        )
        np.testing.assert_allclose(
            engineered["npk_ratio"].values,
            expected.values,
            rtol=1e-5,
        )

    def test_temp_humidity_index_computed_correctly(self, synthetic_df: pd.DataFrame) -> None:
        """feature_engineering_filter computes temp_humidity_index = T*H/100."""
        cleaned: pd.DataFrame = cleaning_filter(synthetic_df.copy())
        engineered: pd.DataFrame = feature_engineering_filter(cleaned)

        expected = cleaned["temperature"] * cleaned["humidity"] / 100.0
        np.testing.assert_allclose(
            engineered["temp_humidity_index"].values,
            expected.values,
            rtol=1e-5,
        )

    def test_one_hot_encoding_adds_crop_type_cols(self, synthetic_df: pd.DataFrame) -> None:
        """feature_engineering_filter one-hot encodes 'label' into crop_type_* cols."""
        cleaned: pd.DataFrame = cleaning_filter(synthetic_df.copy())
        engineered: pd.DataFrame = feature_engineering_filter(cleaned)

        crop_type_cols: list[str] = [c for c in engineered.columns if c.startswith("crop_type_")]
        assert len(crop_type_cols) == len(CROP_TYPES_SUBSET), (
            f"Expected {len(CROP_TYPES_SUBSET)} crop_type_* cols, got {len(crop_type_cols)}"
        )

    def test_original_label_col_dropped(self, synthetic_df: pd.DataFrame) -> None:
        """feature_engineering_filter drops the original 'label' column."""
        cleaned: pd.DataFrame = cleaning_filter(synthetic_df.copy())
        engineered: pd.DataFrame = feature_engineering_filter(cleaned)
        assert "label" not in engineered.columns

    def test_disease_label_column_created(self, synthetic_df: pd.DataFrame) -> None:
        """feature_engineering_filter creates 'disease_label' column."""
        cleaned: pd.DataFrame = cleaning_filter(synthetic_df.copy())
        engineered: pd.DataFrame = feature_engineering_filter(cleaned)
        assert "disease_label" in engineered.columns
        assert set(engineered["disease_label"].unique()) == set(CROP_TYPES_SUBSET)

    def test_yield_kg_per_ha_column_created(self, synthetic_df: pd.DataFrame) -> None:
        """feature_engineering_filter creates 'yield_kg_per_ha' column."""
        cleaned: pd.DataFrame = cleaning_filter(synthetic_df.copy())
        engineered: pd.DataFrame = feature_engineering_filter(cleaned)
        assert "yield_kg_per_ha" in engineered.columns
        assert (engineered["yield_kg_per_ha"] > 0).all()

    def test_no_nan_in_output(self, synthetic_df: pd.DataFrame) -> None:
        """feature_engineering_filter introduces no NaN values."""
        cleaned: pd.DataFrame = cleaning_filter(synthetic_df.copy())
        engineered: pd.DataFrame = feature_engineering_filter(cleaned)
        assert not engineered.isna().any().any()


# ---------------------------------------------------------------------------
# scaling_filter tests
# ---------------------------------------------------------------------------


class TestScalingFilter:
    def test_x_scaled_correct_shape(self, synthetic_df: pd.DataFrame) -> None:
        """scaling_filter returns X_scaled with (n_samples, n_features) shape."""
        cleaned: pd.DataFrame = cleaning_filter(synthetic_df.copy())
        engineered: pd.DataFrame = feature_engineering_filter(cleaned)
        X_scaled, y_disease, y_yield, scaler = scaling_filter(engineered)

        assert X_scaled.shape[0] == len(cleaned)
        assert X_scaled.ndim == 2

    def test_no_nan_in_x_scaled(self, synthetic_df: pd.DataFrame) -> None:
        """scaling_filter produces X_scaled with no NaN values."""
        cleaned: pd.DataFrame = cleaning_filter(synthetic_df.copy())
        engineered: pd.DataFrame = feature_engineering_filter(cleaned)
        X_scaled, _, _, _ = scaling_filter(engineered)
        assert not np.isnan(X_scaled).any()

    def test_no_inf_in_x_scaled(self, synthetic_df: pd.DataFrame) -> None:
        """scaling_filter produces X_scaled with no Inf values."""
        cleaned: pd.DataFrame = cleaning_filter(synthetic_df.copy())
        engineered: pd.DataFrame = feature_engineering_filter(cleaned)
        X_scaled, _, _, _ = scaling_filter(engineered)
        assert not np.isinf(X_scaled).any()

    def test_y_disease_correct_length(self, synthetic_df: pd.DataFrame) -> None:
        """scaling_filter returns y_disease with same length as X_scaled rows."""
        cleaned: pd.DataFrame = cleaning_filter(synthetic_df.copy())
        engineered: pd.DataFrame = feature_engineering_filter(cleaned)
        X_scaled, y_disease, y_yield, scaler = scaling_filter(engineered)
        assert len(y_disease) == X_scaled.shape[0]

    def test_y_yield_correct_length(self, synthetic_df: pd.DataFrame) -> None:
        """scaling_filter returns y_yield with same length as X_scaled rows."""
        cleaned: pd.DataFrame = cleaning_filter(synthetic_df.copy())
        engineered: pd.DataFrame = feature_engineering_filter(cleaned)
        X_scaled, y_disease, y_yield, scaler = scaling_filter(engineered)
        assert len(y_yield) == X_scaled.shape[0]

    def test_returns_four_tuple(self, synthetic_df: pd.DataFrame) -> None:
        """scaling_filter returns a 4-tuple (X_scaled, y_disease, y_yield, scaler)."""
        cleaned: pd.DataFrame = cleaning_filter(synthetic_df.copy())
        engineered: pd.DataFrame = feature_engineering_filter(cleaned)
        result = scaling_filter(engineered)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# training_filter_disease tests
# ---------------------------------------------------------------------------


class TestTrainingFilterDisease:
    def test_returns_fitted_random_forest_classifier(
        self, mini_pipeline_ctx: PipelineContext
    ) -> None:
        """training_filter_disease returns a fitted RandomForestClassifier."""
        from sklearn.ensemble import RandomForestClassifier

        ctx: PipelineContext = mini_pipeline_ctx
        model = training_filter_disease(ctx.X_scaled, ctx.y_disease)
        assert isinstance(model, RandomForestClassifier)

    def test_model_has_predict_method(self, mini_pipeline_ctx: PipelineContext) -> None:
        """Returned disease model has a callable predict method."""
        ctx: PipelineContext = mini_pipeline_ctx
        model = training_filter_disease(ctx.X_scaled, ctx.y_disease)
        assert callable(getattr(model, "predict", None))

    def test_model_predict_returns_correct_shape(
        self, mini_pipeline_ctx: PipelineContext
    ) -> None:
        """Disease model predict returns an array of the same length as input."""
        ctx: PipelineContext = mini_pipeline_ctx
        model = training_filter_disease(ctx.X_scaled, ctx.y_disease)
        preds = model.predict(ctx.X_scaled[:10])
        assert len(preds) == 10

    def test_quality_gate_raises_value_error_for_bad_data(self) -> None:
        """training_filter_disease raises ValueError when accuracy < 0.70.

        Uses 10 randomly shuffled classes with pure noise features so the
        model cannot learn any signal and accuracy stays near 10%.
        """
        rng = np.random.default_rng(999)
        X_noise: np.ndarray = rng.standard_normal((200, 10))
        classes: list[str] = [f"class{i}" for i in range(10)]
        y_random: pd.Series = pd.Series(np.tile(classes, 20))
        rng.shuffle(X_noise)

        with pytest.raises(ValueError):
            training_filter_disease(X_noise, y_random)


# ---------------------------------------------------------------------------
# training_filter_yield tests
# ---------------------------------------------------------------------------


class TestTrainingFilterYield:
    def test_returns_fitted_gradient_boosting_regressor(
        self, mini_pipeline_ctx: PipelineContext
    ) -> None:
        """training_filter_yield returns a fitted GradientBoostingRegressor."""
        from sklearn.ensemble import GradientBoostingRegressor

        ctx: PipelineContext = mini_pipeline_ctx
        model = training_filter_yield(ctx.X_scaled, ctx.y_yield)
        assert isinstance(model, GradientBoostingRegressor)

    def test_model_has_predict_method(self, mini_pipeline_ctx: PipelineContext) -> None:
        """Returned yield model has a callable predict method."""
        ctx: PipelineContext = mini_pipeline_ctx
        model = training_filter_yield(ctx.X_scaled, ctx.y_yield)
        assert callable(getattr(model, "predict", None))

    def test_model_predict_returns_correct_shape(
        self, mini_pipeline_ctx: PipelineContext
    ) -> None:
        """Yield model predict returns an array of the same length as input."""
        ctx: PipelineContext = mini_pipeline_ctx
        model = training_filter_yield(ctx.X_scaled, ctx.y_yield)
        preds = model.predict(ctx.X_scaled[:10])
        assert len(preds) == 10

    def test_quality_gate_raises_value_error_for_bad_data(self) -> None:
        """training_filter_yield raises ValueError when RMSE exceeds 500 kg/ha.

        Uses extreme variance in y (range +-1 000 000) so RMSE vastly
        exceeds the 500 kg/ha quality gate.
        """
        rng = np.random.default_rng(42)
        X_noise: np.ndarray = rng.standard_normal((200, 10))
        y_extreme: pd.Series = pd.Series(rng.uniform(-1_000_000, 1_000_000, 200))

        with pytest.raises(ValueError):
            training_filter_yield(X_noise, y_extreme)


# ---------------------------------------------------------------------------
# output_filter tests
# ---------------------------------------------------------------------------


class TestOutputFilter:
    def test_pkl_files_created(
        self, mini_pipeline_ctx: PipelineContext, tmp_path
    ) -> None:
        """output_filter writes disease_model.pkl, yield_model.pkl, scaler.pkl."""
        output_dir: str = str(tmp_path / "models_out")
        output_filter(mini_pipeline_ctx, output_dir)

        assert os.path.isfile(os.path.join(output_dir, "disease_model.pkl"))
        assert os.path.isfile(os.path.join(output_dir, "yield_model.pkl"))
        assert os.path.isfile(os.path.join(output_dir, "scaler.pkl"))

    def test_pkl_files_not_empty(
        self, mini_pipeline_ctx: PipelineContext, tmp_path
    ) -> None:
        """output_filter writes non-empty .pkl files."""
        output_dir: str = str(tmp_path / "models_size_check")
        output_filter(mini_pipeline_ctx, output_dir)

        for filename in ("disease_model.pkl", "yield_model.pkl", "scaler.pkl"):
            path: str = os.path.join(output_dir, filename)
            assert os.path.getsize(path) > 0, f"{filename} is empty"

    def test_metrics_dict_has_correct_keys(
        self, mini_pipeline_ctx: PipelineContext, tmp_path
    ) -> None:
        """output_filter returns a dict with disease_accuracy, disease_f1_macro, yield_rmse."""
        output_dir: str = str(tmp_path / "models_metrics")
        metrics: dict = output_filter(mini_pipeline_ctx, output_dir)

        assert isinstance(metrics, dict)
        assert "disease_accuracy" in metrics
        assert "disease_f1_macro" in metrics
        assert "yield_rmse" in metrics

    def test_metrics_values_in_valid_ranges(
        self, mini_pipeline_ctx: PipelineContext, tmp_path
    ) -> None:
        """output_filter returns sensible metric value ranges."""
        output_dir: str = str(tmp_path / "models_metric_values")
        metrics: dict = output_filter(mini_pipeline_ctx, output_dir)

        assert 0.0 <= metrics["disease_accuracy"] <= 1.0
        assert 0.0 <= metrics["disease_f1_macro"] <= 1.0
        assert metrics["yield_rmse"] >= 0.0

    def test_creates_output_dir_if_missing(
        self, mini_pipeline_ctx: PipelineContext, tmp_path
    ) -> None:
        """output_filter creates output_dir if it does not exist yet."""
        output_dir: str = str(tmp_path / "new_dir" / "nested_dir")
        assert not os.path.exists(output_dir)

        output_filter(mini_pipeline_ctx, output_dir)
        assert os.path.isdir(output_dir)
