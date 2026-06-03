"""
============================================================================
FEATURE ENGINEERING MODULE
============================================================================
Transforms raw API input into model-ready feature vectors.
Matches EXACTLY the feature engineering from phase2_pipeline.py training.
"""

import numpy as np
from typing import Any


# ── Crop Market Prices (₹ per quintal, approximate MSP / market rates) ──────
CROP_PRICES = {
    "Arhar":      6600,
    "Black gram": 6950,
    "Green gram": 7755,
    "Groundnut":  5850,
    "Horse gram": 4500,
    "Jute":       5050,
    "Maize":      2090,
    "Mustard":    5650,
    "Niger":      7734,
    "Ragi":       3846,
    "Rice":       2183,
    "Sesamum":    7830,
    "Sugarcane":  315,
}

# ── Crop Category Mapping ────────────────────────────────────────────────────
CROP_CATEGORY = {
    "Arhar":      "Pulse",
    "Black gram": "Pulse",
    "Green gram": "Pulse",
    "Groundnut":  "Oilseed",
    "Horse gram": "Pulse",
    "Jute":       "Cash Crop",
    "Maize":      "Cereal",
    "Mustard":    "Oilseed",
    "Niger":      "Oilseed",
    "Ragi":       "Cereal",
    "Rice":       "Cereal",
    "Sesamum":    "Oilseed",
    "Sugarcane":  "Cash Crop",
}

# ── Factor Mappings (matching training pipeline exactly) ─────────────────────

def compute_humidity_factor(humidity: float) -> float:
    if humidity < 47:
        return 1.15
    elif humidity > 85:
        return 0.85
    return 1.0


def compute_irrigation_factor(irrigation: str) -> float:
    mapping = {"Rainfed": 1.20, "Canal": 0.95, "Tube Well": 0.85}
    return mapping.get(irrigation, 0.95)


def compute_season_factor(season: str) -> float:
    mapping = {"Kharif": 0.9, "Rabi": 1.0, "Zaid": 1.15}
    return mapping.get(season, 1.0)


def compute_temp_factor(temperature: float) -> float:
    return 1.15 if temperature > 35.0 else 1.0


def compute_rainfall_factor(rainfall: float) -> float:
    return 0.9 if rainfall > 1500.0 else 1.0


def compute_rainfall_score(rainfall: float) -> int:
    if rainfall > 1500:
        return 4   # Heavy
    elif rainfall >= 1200:
        return 3   # Good
    return 2       # Moderate


def compute_weather_score(rainfall: float, temperature: float) -> int:
    if rainfall > 1500 and temperature < 30:
        return 4   # Cool & Wet
    elif rainfall < 900 and temperature > 33:
        return 1   # Hot & Dry
    elif rainfall >= 1200:
        return 3   # Balanced
    return 2       # Normal


def compute_rainfall_category(rainfall: float) -> str:
    if rainfall < 800:
        return "Low"
    elif rainfall <= 1200:
        return "Medium"
    return "High"


def compute_temperature_category(temperature: float) -> str:
    if temperature < 20:
        return "Low"
    elif temperature <= 35:
        return "Optimal"
    return "High"


def compute_soil_fertility_index(nitrogen: float, phosphorus: float, potassium: float) -> float:
    """Geometric-mean-style SFI matching training data pattern."""
    n_norm = nitrogen / 450.0
    p_norm = phosphorus / 300.0
    k_norm = potassium / 400.0
    return float(np.clip((n_norm * p_norm * k_norm) ** (1 / 3), 0.0, 1.0))


def compute_wsi_raw(temperature: float, rainfall: float) -> float:
    return temperature / (rainfall + 1.0)


def compute_water_stress_index(
    wsi_raw: float,
    humidity_factor: float,
    irrigation_factor: float,
    season_factor: float,
    temp_factor: float,
    rainfall_factor: float,
) -> float:
    return wsi_raw * humidity_factor * irrigation_factor * season_factor * temp_factor * rainfall_factor


def compute_water_stress_level(wsi: float) -> str:
    if wsi >= 0.045:
        return "High Stress"
    elif wsi >= 0.020:
        return "Medium Stress"
    return "Low Stress"


def compute_weather_forecast(rainfall: float, temperature: float) -> str:
    if rainfall > 1500 and temperature < 30:
        return "Cool & Wet"
    elif rainfall < 900 and temperature > 33:
        return "Hot & Dry"
    elif rainfall >= 1200:
        return "Balanced"
    return "Normal"


# ── Encode Categoricals ──────────────────────────────────────────────────────

def safe_label_encode(label_encoders: dict, column: str, value: Any) -> int:
    """Encode a value using saved LabelEncoder, defaulting to 0 for unknowns."""
    if column not in label_encoders:
        return 0
    le = label_encoders[column]
    if value in le.classes_:
        return int(le.transform([value])[0])
    return 0


# ── Build Complete Feature Vector ────────────────────────────────────────────

def engineer_features(raw: dict, label_encoders: dict) -> dict:
    """
    Takes raw API input and returns a flat dict with ALL features
    needed by both yield and crop models.
    """
    # ── Extract raw inputs with sensible defaults ───────────────────
    rainfall    = raw.get("Rainfall", 1000.0)
    temperature = raw.get("Temperature", 30.0)
    nitrogen    = raw.get("Nitrogen", 200.0)
    phosphorus  = raw.get("Phosphorus", 150.0)
    potassium   = raw.get("Potassium", 200.0)
    humidity    = raw.get("Humidity", 70.0)
    soil_ph     = raw.get("Soil_pH", 6.5)
    area        = raw.get("Area", 10000.0)
    fertilizer  = raw.get("Fertilizer", 50.0)
    prev_yield  = raw.get("Previous_Yield", 1.0)
    year        = raw.get("Year", 2025)
    season      = raw.get("Season", "Kharif")
    irrigation  = raw.get("Irrigation", "Rainfed")
    district    = raw.get("District", "Cuttack")
    state       = raw.get("State", "Odisha")
    soil_type   = raw.get("Soil_Type", "Loamy")

    # ── Compute all derived features ────────────────────────────────
    humidity_factor   = compute_humidity_factor(humidity)
    irrigation_factor = compute_irrigation_factor(irrigation)
    season_factor     = compute_season_factor(season)
    temp_factor       = compute_temp_factor(temperature)
    rainfall_factor   = compute_rainfall_factor(rainfall)
    rainfall_score    = compute_rainfall_score(rainfall)
    weather_score     = compute_weather_score(rainfall, temperature)
    soil_fertility    = compute_soil_fertility_index(nitrogen, phosphorus, potassium)
    wsi_raw           = compute_wsi_raw(temperature, rainfall)
    wsi               = compute_water_stress_index(
                            wsi_raw, humidity_factor, irrigation_factor,
                            season_factor, temp_factor, rainfall_factor)
    wsi_level         = compute_water_stress_level(wsi)
    weather_forecast  = compute_weather_forecast(rainfall, temperature)

    yield_change      = 0.0  # Default — no historical info at prediction time
    yield_trend       = prev_yield + yield_change

    # ── Ordinal encodings ───────────────────────────────────────────
    rainfall_dist_map = {"Low": 0, "Moderate": 1, "Good": 2, "Heavy": 3}
    wsi_level_map     = {"Low Stress": 0, "Medium Stress": 1, "High Stress": 2}

    rainfall_dist = "Moderate"
    if rainfall > 1500:
        rainfall_dist = "Heavy"
    elif rainfall >= 1200:
        rainfall_dist = "Good"

    rainfall_dist_encoded    = rainfall_dist_map.get(rainfall_dist, 1)
    water_stress_lvl_encoded = wsi_level_map.get(wsi_level, 1)

    # ── Crop category for the input (use Pulse as default) ──────────
    crop_category = raw.get("Crop_Category", "Pulse")

    # ── Label-encoded categoricals ──────────────────────────────────
    district_encoded       = safe_label_encode(label_encoders, "District", district)
    state_encoded          = safe_label_encode(label_encoders, "State", state)
    soil_type_encoded      = safe_label_encode(label_encoders, "Soil_Type", soil_type)
    irrigation_encoded     = safe_label_encode(label_encoders, "Irrigation", irrigation)
    season_encoded         = safe_label_encode(label_encoders, "Season", season)
    crop_category_encoded  = safe_label_encode(label_encoders, "Crop_Category", crop_category)
    weather_forecast_enc   = safe_label_encode(label_encoders, "Weather_Forecast", weather_forecast)

    # ── Build the full feature dict ─────────────────────────────────
    features = {
        # Raw numeric
        "Humidity":              humidity,
        "Soil_pH":               soil_ph,
        "Temperature":           temperature,
        "Rainfall":              rainfall,
        "Nitrogen":              nitrogen,
        "Phosphorus":            phosphorus,
        "Potassium":             potassium,
        "Area":                  area,
        "Fertilizer_kg_per_ha":  fertilizer,
        "Previous_Yield":        prev_yield,
        "Year":                  year,

        # Derived scores
        "Rainfall_Score":        rainfall_score,
        "Weather_Score":         weather_score,
        "Soil_Fertility_Index":  soil_fertility,
        "WSI_raw":               wsi_raw,
        "Water_Stress_Index":    wsi,
        "Yield_Change":          yield_change,
        "Yield_Trend":           yield_trend,

        # Factors
        "Humidity_Factor":       humidity_factor,
        "Irrigation_Factor":     irrigation_factor,
        "Season_Factor":         season_factor,
        "Temp_Factor":           temp_factor,
        "Rainfall_Factor":       rainfall_factor,

        # Encoded categoricals
        "District_encoded":      district_encoded,
        "State_encoded":         state_encoded,
        "Soil_Type_encoded":     soil_type_encoded,
        "Irrigation_encoded":    irrigation_encoded,
        "Season_encoded":        season_encoded,
        "Crop_Category_encoded": crop_category_encoded,
        "Weather_Forecast_encoded": weather_forecast_enc,

        # Ordinal encoded
        "Rainfall_Distribution_encoded": rainfall_dist_encoded,
        "Water_Stress_Level_encoded":    water_stress_lvl_encoded,
    }

    return features


def generate_insights(raw: dict) -> list[str]:
    """Generate smart agricultural insights from raw input."""
    insights = []
    temperature = raw.get("Temperature", 30)
    rainfall    = raw.get("Rainfall", 1000)
    humidity    = raw.get("Humidity", 70)
    soil_ph     = raw.get("Soil_pH", 6.5)
    nitrogen    = raw.get("Nitrogen", 200)
    phosphorus  = raw.get("Phosphorus", 150)
    potassium   = raw.get("Potassium", 200)
    fertilizer  = raw.get("Fertilizer", 50)
    prev_yield  = raw.get("Previous_Yield", 1.0)

    # ── Temperature insights ────────────────────────────────────────
    if temperature > 38:
        insights.append("🔴 Severe heat stress risk — consider heat-tolerant varieties and mulching")
    elif temperature > 35:
        insights.append("🟠 High temperature may reduce yield by 10–20% — schedule irrigation during cooler hours")
    elif temperature < 15:
        insights.append("🔵 Cold stress detected — consider frost protection for sensitive crops")

    # ── Rainfall insights ───────────────────────────────────────────
    if rainfall < 600:
        insights.append("🔴 Critical drought conditions — irrigated cultivation mandatory")
    elif rainfall < 800:
        insights.append("🟠 Low rainfall detected → supplemental irrigation strongly recommended")
    elif rainfall > 1800:
        insights.append("🟠 Excessive rainfall risk — ensure proper drainage and choose flood-tolerant varieties")
    elif rainfall > 1500:
        insights.append("🟡 High rainfall — monitor for waterlogging in low-lying areas")

    # ── Soil pH insights ────────────────────────────────────────────
    if soil_ph < 5.0:
        insights.append("🔴 Soil is strongly acidic (pH < 5) — lime application essential before sowing")
    elif soil_ph < 5.5:
        insights.append("🟠 Soil is acidic (pH < 5.5) — consider liming to improve nutrient availability")
    elif soil_ph > 8.5:
        insights.append("🔴 Soil is highly alkaline — gypsum application recommended")
    elif soil_ph > 8.0:
        insights.append("🟠 Soil is alkaline — may limit micronutrient uptake")

    # ── Humidity insights ───────────────────────────────────────────
    if humidity < 40:
        insights.append("🟠 Dry climate conditions — increase irrigation frequency, consider drip irrigation")
    elif humidity > 90:
        insights.append("🟡 Very high humidity — increased risk of fungal diseases; plan preventive spraying")

    # ── Nutrient insights ───────────────────────────────────────────
    sfi = (nitrogen + phosphorus + potassium) / 3
    if sfi > 300:
        insights.append("✅ Soil fertility is excellent — maintain current nutrient management")
    elif sfi > 200:
        insights.append("🟡 Soil fertility is moderate — consider balanced NPK supplementation")
    elif sfi > 100:
        insights.append("🟠 Soil fertility is low — increase fertilizer application with soil test guidance")
    else:
        insights.append("🔴 Soil fertility is critically low — intensive nutrient management needed")

    # ── NPK balance ─────────────────────────────────────────────────
    n_p_ratio = nitrogen / (phosphorus + 1)
    if n_p_ratio > 3:
        insights.append("🟡 Nitrogen-heavy nutrient profile — increase phosphorus for better root development")
    elif n_p_ratio < 0.8:
        insights.append("🟡 Low nitrogen relative to phosphorus — increase urea application")

    # ── Fertilizer insights ─────────────────────────────────────────
    if fertilizer < 25:
        insights.append("🟠 Very low fertilizer input — yields will be significantly below potential")
    elif fertilizer > 300:
        insights.append("🟡 High fertilizer usage — diminishing returns likely; optimize for cost-efficiency")

    # ── Previous yield trend ────────────────────────────────────────
    if prev_yield < 0.5:
        insights.append("🟡 Previous yield was very low — investigate soil health and pest history")

    # ── Ensure at least one insight ─────────────────────────────────
    if not insights:
        insights.append("✅ Growing conditions appear favorable for this region and season")

    return insights


def estimate_profit(crop: str, yield_value: float, area: float) -> float:
    """
    Estimate profit in ₹.
    Yield is in tonnes/ha, Area in hectares.
    Profit = Revenue - Cost. Cost ≈ 40% of revenue for simplification.
    """
    price_per_quintal = CROP_PRICES.get(crop, 3000)
    # Yield (tonnes/ha) → quintals/ha = yield * 10
    quintals = yield_value * 10.0
    revenue_per_ha = quintals * price_per_quintal
    cost_per_ha = revenue_per_ha * 0.40  # Approximate cost ratio
    profit_per_ha = revenue_per_ha - cost_per_ha
    # Total profit for given area (Area in the dataset is in hectares)
    total_profit = profit_per_ha * (area / 10000.0)  # Normalize if area is in raw units
    return round(max(total_profit, 0), 2)
