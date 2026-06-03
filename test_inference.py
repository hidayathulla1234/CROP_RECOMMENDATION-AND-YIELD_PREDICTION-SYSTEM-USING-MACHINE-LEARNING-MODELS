import joblib
import numpy as np
from app.main import build_model_vector, safe_encode_crop_category
from app.utils.feature_engineering import engineer_features, CROP_CATEGORY

models = {}
for k in ["best_crop_model.pkl", "crop_scaler.pkl", "crop_features.pkl", "label_encoders.pkl"]:
    models[k.split(".")[0]] = joblib.load("models/" + k)

raw = {"District":"Bolangir","State":"Odisha","Season":"Zaid","Soil_Type":"Red","Irrigation":"Rainfed","Rainfall":500,"Temperature":38,"Humidity":40,"Nitrogen":150,"Phosphorus":100,"Potassium":110,"Soil_pH":5.8,"Area":10000,"Fertilizer":50,"Previous_Yield":1.5,"Year":2026}

le_crops = models["label_encoders"]["Crops"]
features = engineer_features(raw, models["label_encoders"])
print(features)
X_crop = build_model_vector(features, models["crop_features"], models["crop_scaler"])

proba = models["best_crop_model"].predict_proba(X_crop)[0]
res = []
for i, p in enumerate(proba):
    crop_name = le_crops.inverse_transform([i])[0]
    res.append((crop_name, p))

res.sort(key=lambda x: -x[1])
print("TOP 3:", res[:3])
