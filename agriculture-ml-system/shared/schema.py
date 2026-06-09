from __future__ import annotations

from pydantic import BaseModel


class CropFeaturesInput(BaseModel):
    """Raw crop sensor/survey features submitted by a client."""

    nitrogen: float       # N content in kg/ha
    phosphorus: float     # P content in kg/ha
    potassium: float      # K content in kg/ha
    temperature: float    # Mean temperature in Celsius
    humidity: float       # Relative humidity percentage 0-100
    ph: float             # Soil pH 0-14
    rainfall: float       # Annual rainfall in mm
    crop_type: str        # One of the 22 known crop types


class PredictionResponse(BaseModel):
    """Unified response returned by the API Gateway after a /predict call."""

    disease_label: str
    disease_confidence: float
    predicted_yield_kg_per_ha: float
    log_id: str


class ScaledFeatures(BaseModel):
    """Flat list of z-score normalised floats produced by the Preprocessing Service."""

    features: list[float]


class DiseaseOutput(BaseModel):
    """Response from the Disease Detection Service."""

    disease_label: str
    confidence: float


class YieldOutput(BaseModel):
    """Response from the Yield Prediction Service."""

    predicted_yield_kg_per_ha: float


class PredictionLog(BaseModel):
    """Payload POSTed to the Logging Service for each prediction event."""

    features: list[float]
    disease_label: str
    disease_confidence: float
    predicted_yield_kg_per_ha: float
    timestamp: str   # ISO 8601 formatted datetime string


class LogResponse(BaseModel):
    """Response returned by the Logging Service after storing a log entry."""

    message: str
    log_id: str
