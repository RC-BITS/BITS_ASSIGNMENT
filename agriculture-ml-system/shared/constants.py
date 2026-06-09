from __future__ import annotations

# ---------------------------------------------------------------------------
# Crop vocabulary
# ---------------------------------------------------------------------------

CROP_TYPES: list[str] = [
    "rice",
    "wheat",
    "maize",
    "chickpea",
    "kidneybeans",
    "pigeonpeas",
    "mothbeans",
    "mungbean",
    "blackgram",
    "lentil",
    "pomegranate",
    "banana",
    "mango",
    "grapes",
    "watermelon",
    "muskmelon",
    "apple",
    "orange",
    "papaya",
    "coconut",
    "cotton",
    "jute",
    "coffee",
]

# Disease labels reuse the same 22-class vocabulary as the crop recommendation
# dataset (one disease-proxy label per crop class).
DISEASE_LABELS: list[str] = CROP_TYPES.copy()

# ---------------------------------------------------------------------------
# Feature column definitions
# ---------------------------------------------------------------------------

NUMERIC_COLS: list[str] = [
    "nitrogen",
    "phosphorus",
    "potassium",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]

# Columns where a value of 0 is physically impossible and should be treated as
# missing (replaced with NaN before median imputation).
ZERO_INVALID_COLS: list[str] = [
    "phosphorus",
    "potassium",
    "temperature",
    "humidity",
    "ph",
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODELS_DIR: str = "./models"

DATASET_URL: str = (
    "https://raw.githubusercontent.com/AtharvaMusale/"
    "Crop-Recommendation-Dataset/main/Crop_recommendation.csv"
)
