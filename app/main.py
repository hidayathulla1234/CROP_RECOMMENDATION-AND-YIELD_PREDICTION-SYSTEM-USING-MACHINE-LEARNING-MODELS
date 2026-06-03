"""
============================================================================
CROP RECOMMENDATION & YIELD PREDICTION API
============================================================================
Production-ready FastAPI backend.

Run:  uvicorn app.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""

import os
import sys
import time
import logging
import warnings
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from app.utils.feature_engineering import (
    engineer_features,
    generate_insights,
    estimate_profit,
    CROP_PRICES,
    CROP_CATEGORY,
)

warnings.filterwarnings("ignore")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crop_api")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── Global model store ───────────────────────────────────────────────────────
models: dict = {}


def load_models():
    """Load all ML artifacts into memory."""
    artifacts = {
        "yield_model":    "best_yield_model.pkl",
        "yield_scaler":   "yield_scaler.pkl",
        "yield_features": "yield_features.pkl",
        "crop_model":     "best_crop_model.pkl",
        "crop_scaler":    "crop_scaler.pkl",
        "crop_features":  "crop_features.pkl",
        "label_encoders": "label_encoders.pkl",
    }

    # Also try loading the user-specified filename as a fallback
    climate_model_path = os.path.join(MODELS_DIR, "climate_driven_model.pkl")
    if os.path.exists(climate_model_path):
        models["climate_model"] = joblib.load(climate_model_path)
        logger.info("✅ Loaded climate_driven_model.pkl")

    for key, fname in artifacts.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            models[key] = joblib.load(path)
            logger.info(f"✅ Loaded {fname}")
        else:
            logger.warning(f"⚠️  {fname} not found at {path}")

    # Validate critical models
    required = ["yield_model", "crop_model", "label_encoders"]
    missing = [k for k in required if k not in models]
    if missing:
        logger.error(f"❌ Missing critical models: {missing}")
        logger.error("   Run phase2_pipeline.py first to train models.")
    else:
        logger.info("🚀 All models loaded successfully")


# ── Application Lifespan ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    logger.info("=" * 60)
    logger.info("  CROP RECOMMENDATION API — Starting Up")
    logger.info("=" * 60)
    load_models()
    yield
    logger.info("🛑 Shutting down API")
    models.clear()


# ── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="🌾 Crop Recommendation & Yield Prediction API",
    description=(
        "Production-ready ML API for predicting crop yield and recommending "
        "optimal crops based on climate, soil, and farming conditions. "
        "Trained on Odisha agricultural data (2015–2025)."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Schemas ───────────────────────────────────────────────

class PredictionInput(BaseModel):
    """Input schema for crop prediction."""
    District:       str   = Field(default="Cuttack", description="District name in Odisha")
    State:          str   = Field(default="Odisha", description="State (currently Odisha only)")
    Season:         str   = Field(default="Kharif", description="Season: Kharif, Rabi, or Zaid")
    Soil_Type:      str   = Field(default="Loamy", description="Soil type: Alluvial, Black, Laterite, Loamy, Red, Sandy")
    Irrigation:     str   = Field(default="Rainfed", description="Irrigation: Canal, Rainfed, or Tube Well")

    Rainfall:       float = Field(default=1000.0, ge=0, le=3000, description="Annual rainfall in mm")
    Temperature:    float = Field(default=30.0, ge=0, le=50, description="Average temperature in °C")
    Nitrogen:       float = Field(default=200.0, ge=0, le=500, description="Nitrogen content (kg/ha)")
    Phosphorus:     float = Field(default=150.0, ge=0, le=400, description="Phosphorus content (kg/ha)")
    Potassium:      float = Field(default=200.0, ge=0, le=500, description="Potassium content (kg/ha)")
    Humidity:       float = Field(default=70.0, ge=0, le=100, description="Relative humidity (%)")
    Soil_pH:        float = Field(default=6.5, ge=3.0, le=9.0, description="Soil pH level")
    Area:           float = Field(default=10000.0, ge=1, description="Cultivated area (hectares)")
    Fertilizer:     float = Field(default=50.0, ge=0, le=500, description="Fertilizer applied (kg/ha)")
    Previous_Yield: float = Field(default=1.0, ge=0, description="Previous season yield (tonnes/ha)")
    Year:           int   = Field(default=2026, ge=2000, le=2050, description="Prediction year")

    @field_validator("Season")
    @classmethod
    def validate_season(cls, v):
        valid = ["Kharif", "Rabi", "Zaid"]
        if v not in valid:
            raise ValueError(f"Season must be one of {valid}")
        return v

    @field_validator("Soil_Type")
    @classmethod
    def validate_soil(cls, v):
        valid = ["Alluvial", "Black", "Laterite", "Loamy", "Red", "Sandy"]
        if v not in valid:
            raise ValueError(f"Soil_Type must be one of {valid}")
        return v

    @field_validator("Irrigation")
    @classmethod
    def validate_irrigation(cls, v):
        valid = ["Canal", "Rainfed", "Tube Well"]
        if v not in valid:
            raise ValueError(f"Irrigation must be one of {valid}")
        return v

    model_config = {"json_schema_extra": {
        "examples": [{
            "District": "Dhenkanal",
            "State": "Odisha",
            "Season": "Rabi",
            "Soil_Type": "Red",
            "Irrigation": "Rainfed",
            "Rainfall": 750,
            "Temperature": 40,
            "Nitrogen": 250,
            "Phosphorus": 159,
            "Potassium": 189,
            "Humidity": 55,
            "Soil_pH": 5.0,
            "Area": 10000,
            "Fertilizer": 72,
            "Previous_Yield": 1.5,
            "Year": 2026,
        }],
    }}


class CropRecommendation(BaseModel):
    Crop: str
    Profit: float


class PredictionResponse(BaseModel):
    Best_Crop: str
    Expected_Yield: float
    Estimated_Profit: float
    Top_3_Recommendations: list[CropRecommendation]
    Insights: list[str]
    Feature_Summary: dict


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
    version: str


# ── Helper: build model input ────────────────────────────────────────────────

def build_model_vector(features: dict, feature_list: list, scaler) -> np.ndarray:
    """Build a scaled feature vector in the exact order the model expects."""
    vec = []
    for f in feature_list:
        vec.append(float(features.get(f, 0.0)))
    X = np.array([vec])
    X_scaled = scaler.transform(X)
    return X_scaled


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API status and loaded models."""
    return HealthResponse(
        status="API is running",
        models_loaded={
            "yield_model":    "yield_model"    in models,
            "crop_model":     "crop_model"     in models,
            "label_encoders": "label_encoders" in models,
            "yield_scaler":   "yield_scaler"   in models,
            "crop_scaler":    "crop_scaler"    in models,
        },
        version="2.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(input_data: PredictionInput):
    """
    🌾 **Crop Recommendation & Yield Prediction**

    Takes farming conditions and returns:
    - Best crop recommendation
    - Expected yield (tonnes/ha)
    - Estimated profit (₹)
    - Top 3 crop recommendations with profit estimates
    - Smart agricultural insights
    """
    start = time.time()

    # ── Validate models are loaded ──────────────────────────────────
    required_keys = ["crop_model", "yield_model", "label_encoders",
                     "crop_scaler", "yield_scaler",
                     "crop_features", "yield_features"]
    missing = [k for k in required_keys if k not in models]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Models not loaded: {missing}. Run the training pipeline first.",
        )

    raw = input_data.model_dump()
    label_encoders = models["label_encoders"]

    # ── Step 1: Feature Engineering ─────────────────────────────────
    features = engineer_features(raw, label_encoders)

    # ── Step 2: Crop Recommendation ─────────────────────────────────
    crop_features_list = models["crop_features"]
    crop_scaler        = models["crop_scaler"]
    crop_model         = models["crop_model"]
    le_crops           = label_encoders.get("Crops")

    X_crop = build_model_vector(features, crop_features_list, crop_scaler)

    # Top recommendations — direct multi-class prediction (no leaky Crop_Category feature)
    top_crops = []
    if hasattr(crop_model, "predict_proba"):
        # Build vector without Crop_Category_encoded (removed from crop model features)
        X_crop_direct = build_model_vector(features, crop_features_list, crop_scaler)
        proba = crop_model.predict_proba(X_crop_direct)[0]

        crop_probs_dict = {}
        for i, p in enumerate(proba):
            crop_name = le_crops.inverse_transform([i])[0] if le_crops else str(i)
            crop_probs_dict[crop_name] = float(p)

        crop_probs = sorted(crop_probs_dict.items(), key=lambda x: -x[1])
        top_crops = crop_probs[:5]  # Get top 5 for yield estimation
        best_crop = top_crops[0][0] if top_crops else "Unknown"
    else:
        pred_encoded = crop_model.predict(X_crop)[0]
        best_crop = le_crops.inverse_transform([int(pred_encoded)])[0] if le_crops else str(pred_encoded)
        top_crops = [(best_crop, 1.0)]

    # ── Step 3: Yield Prediction for top crops ──────────────────────
    yield_features_list = models["yield_features"]
    yield_scaler        = models["yield_scaler"]
    yield_model         = models["yield_model"]

    # Typical yield ranges (t/ha) per crop — used to correct CatBoost
    # predictions that may underestimate Cash Crops like Sugarcane
    YIELD_RANGE = {
        "Sugarcane":  (50.0,  100.0),  # Sugarcane: 50–100 t/ha in Odisha
        "Jute":       (2.0,   4.0),
        "Rice":       (2.0,   6.0),
        "Maize":      (2.0,   5.0),
        "Groundnut":  (1.5,   3.5),
        "Arhar":      (0.8,   2.0),
        "Black gram": (0.5,   1.8),
        "Green gram": (0.5,   1.6),
        "Mustard":    (1.0,   2.5),
        "Niger":      (0.3,   1.0),
        "Ragi":       (1.0,   2.5),
        "Sesamum":    (0.4,   1.2),
        "Horse gram": (0.4,   1.1),
    }

    crop_yields = {}
    for crop_name, prob in top_crops:
        # Yield model still uses Crop_Category — set it correctly per crop
        cat = CROP_CATEGORY.get(crop_name, "Pulse")
        features["Crop_Category_encoded"] = safe_encode_crop_category(
            label_encoders, cat
        )
        X_yield = build_model_vector(features, yield_features_list, yield_scaler)
        predicted_yield = float(yield_model.predict(X_yield)[0])
        predicted_yield = max(predicted_yield, 0.1)  # Floor at 0.1 t/ha

        # Clamp to known realistic range for this crop
        lo, hi = YIELD_RANGE.get(crop_name, (0.1, 200.0))
        predicted_yield = float(np.clip(predicted_yield, lo, hi))
        crop_yields[crop_name] = round(predicted_yield, 4)

    # ── Step 4: Estimate Profits ────────────────────────────────────
    area = raw.get("Area", 10000)
    recommendations = []
    for crop_name, prob in top_crops:
        yld = crop_yields.get(crop_name, 1.0)
        profit = estimate_profit(crop_name, yld, area)
        score = profit * prob  # Expected profit = Economics * Agronomic Viability
        recommendations.append({"Crop": crop_name, "Profit": profit, "Yield": yld, "Probability": round(prob, 4), "Score": score})

    recommendations.sort(key=lambda x: -x["Score"])
    top_3 = recommendations[:3]

    best_entry = recommendations[0] if recommendations else {"Crop": best_crop, "Profit": 0, "Yield": 1.0}

    # ── Step 5: Generate Insights ───────────────────────────────────
    insights = generate_insights(raw)

    # ── Step 6: Feature Summary ─────────────────────────────────────
    from app.utils.feature_engineering import (
        compute_rainfall_category,
        compute_temperature_category,
        compute_soil_fertility_index,
        compute_water_stress_index,
        compute_wsi_raw,
        compute_humidity_factor,
        compute_irrigation_factor,
        compute_season_factor,
        compute_temp_factor,
        compute_rainfall_factor,
    )

    sfi = compute_soil_fertility_index(raw["Nitrogen"], raw["Phosphorus"], raw["Potassium"])
    wsi_r = compute_wsi_raw(raw["Temperature"], raw["Rainfall"])
    wsi = compute_water_stress_index(
        wsi_r,
        compute_humidity_factor(raw["Humidity"]),
        compute_irrigation_factor(raw["Irrigation"]),
        compute_season_factor(raw["Season"]),
        compute_temp_factor(raw["Temperature"]),
        compute_rainfall_factor(raw["Rainfall"]),
    )
    weather_norm = min(
        (raw["Rainfall"] / 2000) * 0.6 + (1 - abs(raw["Temperature"] - 28) / 22) * 0.4,
        1.0,
    )

    feature_summary = {
        "Rainfall_Category":     compute_rainfall_category(raw["Rainfall"]),
        "Temperature_Category":  compute_temperature_category(raw["Temperature"]),
        "Soil_Fertility_Index":  round(sfi, 4),
        "Water_Stress_Index":    round(wsi, 6),
        "Weather_Score":         round(max(weather_norm, 0), 4),
        "Yield_Efficiency":      round(raw["Previous_Yield"] / (sfi + 1), 4),
        "Processing_Time_ms":    round((time.time() - start) * 1000, 1),
    }

    return PredictionResponse(
        Best_Crop=best_entry["Crop"],
        Expected_Yield=best_entry["Yield"],
        Estimated_Profit=best_entry["Profit"],
        Top_3_Recommendations=[
            CropRecommendation(Crop=r["Crop"], Profit=r["Profit"])
            for r in top_3
        ],
        Insights=insights,
        Feature_Summary=feature_summary,
    )


# ── Utility ──────────────────────────────────────────────────────────────────

def safe_encode_crop_category(label_encoders: dict, category: str) -> int:
    le = label_encoders.get("Crop_Category")
    if le is None:
        return 0
    if category in le.classes_:
        return int(le.transform([category])[0])
    return 0


# ── Error Handlers ───────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


# ── Static Files (Serve Frontend) ────────────────────────────────────────────

# This allows you to visit http://127.0.0.1:8000 to see the website
# It MUST be at the end so it doesn't shadow the /predict API route
app.mount("/", StaticFiles(directory=BASE_DIR, html=True), name="static")
