import joblib
from app.utils.feature_engineering import engineer_features, build_model_vector, CROP_CATEGORY, safe_encode_crop_category

models = {}
for k in ["best_crop_model.pkl", "crop_scaler.pkl", "crop_features.pkl", "label_encoders.pkl"]:
    models[k.split(".")[0]] = joblib.load("models/" + k)


raw = {"District":"Bolangir","State":"Odisha","Season":"Zaid","Soil_Type":"Red","Irrigation":"Rainfed","Rainfall":500,"Temperature":38,"Humidity":40,"Nitrogen":150,"Phosphorus":100,"Potassium":110,"Soil_pH":5.8,"Area":10000,"Fertilizer":50,"Previous_Yield":1.5,"Year":2026}

le_crops = models["label_encoders"]["Crops"]
features = engineer_features(raw, models["label_encoders"])

top_crops_candidates = []
for cat in ["Pulse", "Cereal", "Oilseed", "Cash Crop"]:
    features["Crop_Category_encoded"] = safe_encode_crop_category(models["label_encoders"], cat)
    X_crop = build_model_vector(features, models["crop_features"], models["crop_scaler"])
    proba = models["best_crop_model"].predict_proba(X_crop)[0]
    for i, p in enumerate(proba):
        crop_name = le_crops.inverse_transform([i])[0]
        if CROP_CATEGORY.get(crop_name, "Pulse") == cat:
            top_crops_candidates.append((crop_name, p))

top_crops_candidates.sort(key=lambda x: -x[1])
print(top_crops_candidates)
