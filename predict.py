"""
============================================================================
PREDICTION API — Crop Yield & Crop Recommendation
============================================================================
Lightweight inference module. Load saved models and predict.
"""

import os, joblib, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')


def _load_artifacts():
    """Load all saved model artifacts."""
    artifacts = {}
    files = {
        'yield_model': 'best_yield_model.pkl',
        'yield_scaler': 'yield_scaler.pkl',
        'yield_features': 'yield_features.pkl',
        'crop_model': 'best_crop_model.pkl',
        'crop_scaler': 'crop_scaler.pkl',
        'crop_features': 'crop_features.pkl',
        'label_encoders': 'label_encoders.pkl',
    }
    for key, fname in files.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            artifacts[key] = joblib.load(path)
        else:
            print(f"  WARNING: {fname} not found. Run crop_ml_pipeline.py first.")
    return artifacts


# Cache artifacts on first load
_ARTIFACTS = None

def _get_artifacts():
    global _ARTIFACTS
    if _ARTIFACTS is None:
        _ARTIFACTS = _load_artifacts()
    return _ARTIFACTS


def _prepare_input(input_data: dict, required_features: list, scaler, label_encoders: dict):
    """Transform raw input dict into model-ready feature vector."""
    # Start with defaults
    defaults = {
        'Year': 2025, 'Area': 10000, 'Previous_Yield': 1.0, 'Yield_Change': 0.0,
        'Rainfall_Score': 2, 'Weather_Score': 2, 'Soil_Fertility_Index': 0.5,
        'WSI_raw': 0.03, 'Humidity_Factor': 1.0, 'Irrigation_Factor': 0.95,
        'Season_Factor': 1.0, 'Temp_Factor': 1.0, 'Rainfall_Factor': 1.0,
        'Water_Stress_Index': 0.03, 'Fertilizer_kg_per_ha': 50,
        'Yield_Efficiency': 1.0,
    }
    row = {**defaults, **input_data}

    # Encode categoricals
    cat_mappings = {
        'Soil_Type': 'Soil_Type_encoded',
        'Irrigation': 'Irrigation_encoded',
        'Season': 'Season_encoded',
        'District': 'District_encoded',
        'State': 'State_encoded',
        'Crop_Category': 'Crop_Category_encoded',
        'Weather_Forecast': 'Weather_Forecast_encoded',
    }
    for raw_col, enc_col in cat_mappings.items():
        if raw_col in row and raw_col in label_encoders:
            le = label_encoders[raw_col]
            val = row[raw_col]
            if val in le.classes_:
                row[enc_col] = le.transform([val])[0]
            else:
                row[enc_col] = 0

    # Ordinal encoding
    ordinal = {
        'Rainfall_Distribution': {'Low': 0, 'Moderate': 1, 'Good': 2, 'Heavy': 3},
        'Water_Stress_Level': {'Low Stress': 0, 'Medium Stress': 1, 'High Stress': 2},
    }
    for col, mapping in ordinal.items():
        if col in row:
            row[col + '_encoded'] = mapping.get(row[col], 1)

    # Engineered features
    N = row.get('Nitrogen', 200)
    P = row.get('Phosphorus', 150)
    K = row.get('Potassium', 200)
    row['Nutrient_Ratio'] = (N + P + K) / 3
    row['Climate_Index'] = (row.get('Rainfall', 1000) + row.get('Temperature', 30) + row.get('Humidity', 70)) / 3
    row['Soil_Health'] = (row.get('Soil_pH', 6.5) + row.get('Soil_Fertility_Index', 0.5)) / 2
    row['Water_Index'] = (row.get('Water_Stress_Index', 0.03) + row.get('Irrigation_Factor', 0.95)) / 2
    row['Yield_Trend'] = row.get('Previous_Yield', 1.0) + row.get('Yield_Change', 0.0)
    fert = row.get('Fertilizer_kg_per_ha', 50)
    row['Fertilizer_Efficiency'] = row.get('Yield', 1.0) / fert if fert > 0 else 0

    # Build feature vector in correct order
    feature_vec = []
    for f in required_features:
        feature_vec.append(float(row.get(f, 0)))

    X = np.array([feature_vec])
    X_scaled = scaler.transform(X)
    return X_scaled


def predict_crop_yield(input_data: dict) -> dict:
    """
    Predict crop yield given farming conditions.

    Parameters
    ----------
    input_data : dict
        Keys can include: Rainfall, Temperature, Soil_Type, Nitrogen,
        Phosphorus, Potassium, Humidity, Soil_pH, Irrigation, Season,
        Area, Fertilizer_kg_per_ha, Previous_Yield, etc.

    Returns
    -------
    dict with 'predicted_yield' and 'confidence'
    """
    arts = _get_artifacts()
    if 'yield_model' not in arts:
        return {'error': 'Yield model not loaded. Run pipeline first.'}

    X = _prepare_input(input_data, arts['yield_features'],
                       arts['yield_scaler'], arts.get('label_encoders', {}))
    pred = arts['yield_model'].predict(X)[0]

    return {
        'predicted_yield': round(float(pred), 4),
        'unit': 'tonnes/hectare',
        'input': input_data,
    }


def recommend_crop(input_data: dict) -> dict:
    """
    Recommend the best crop given farming conditions.

    Parameters
    ----------
    input_data : dict
        Keys can include: Rainfall, Temperature, Soil_Type, Nitrogen,
        Phosphorus, Potassium, Humidity, Soil_pH, Irrigation, etc.

    Returns
    -------
    dict with 'recommended_crop' and 'probabilities'
    """
    arts = _get_artifacts()
    if 'crop_model' not in arts:
        return {'error': 'Crop model not loaded. Run pipeline first.'}

    X = _prepare_input(input_data, arts['crop_features'],
                       arts['crop_scaler'], arts.get('label_encoders', {}))

    pred_encoded = arts['crop_model'].predict(X)[0]

    le_crops = arts['label_encoders'].get('Crops')
    if le_crops is not None:
        crop_name = le_crops.inverse_transform([int(pred_encoded)])[0]
    else:
        crop_name = str(pred_encoded)

    # Get probabilities if available
    probs = {}
    if hasattr(arts['crop_model'], 'predict_proba'):
        prob_array = arts['crop_model'].predict_proba(X)[0]
        if le_crops is not None:
            for i, p in enumerate(prob_array):
                probs[le_crops.inverse_transform([i])[0]] = round(float(p), 4)
        probs = dict(sorted(probs.items(), key=lambda x: -x[1])[:5])

    return {
        'recommended_crop': crop_name,
        'top_probabilities': probs,
        'input': input_data,
    }


# ============================================================================
# DEMO
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  CROP ML SYSTEM — Prediction Demo")
    print("=" * 60)

    sample_input = {
        'Rainfall': 1200,
        'Temperature': 30,
        'Soil_Type': 'Loamy',
        'Nitrogen': 300,
        'Phosphorus': 200,
        'Potassium': 250,
        'Humidity': 75,
        'Soil_pH': 6.5,
        'Irrigation': 'Canal',
        'Season': 'Kharif',
        'Area': 15000,
        'Fertilizer_kg_per_ha': 80,
    }

    print("\n  Input:")
    for k, v in sample_input.items():
        print(f"    {k}: {v}")

    print("\n  --- Yield Prediction ---")
    result = predict_crop_yield(sample_input)
    if 'error' not in result:
        print(f"  Predicted Yield: {result['predicted_yield']} {result['unit']}")
    else:
        print(f"  {result['error']}")

    print("\n  --- Crop Recommendation ---")
    result = recommend_crop(sample_input)
    if 'error' not in result:
        print(f"  Recommended Crop: {result['recommended_crop']}")
        if result['top_probabilities']:
            print("  Top probabilities:")
            for crop, prob in result['top_probabilities'].items():
                print(f"    {crop}: {prob:.4f}")
    else:
        print(f"  {result['error']}")
