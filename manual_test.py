"""
============================================================================
MANUAL TESTING SYSTEM — Crop Yield & Crop Recommendation
Interactive CLI for real-world agricultural predictions
============================================================================
Usage:
    python3 manual_test.py
"""
import os, sys, warnings, joblib
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '1'

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# ─────────────────────── ANSI colour helpers ────────────────────────────────
GREEN  = '\033[92m'
YELLOW = '\033[93m'
CYAN   = '\033[96m'
BOLD   = '\033[1m'
RED    = '\033[91m'
RESET  = '\033[0m'

def h(text, colour=CYAN):  return f"{colour}{BOLD}{text}{RESET}"
def ok(text):              return f"{GREEN}✅  {text}{RESET}"
def warn(text):            return f"{YELLOW}⚠️   {text}{RESET}"
def err(text):             return f"{RED}❌  {text}{RESET}"

# ─────────────────────── Artifact loading ───────────────────────────────────
_cache = {}

def _load_all():
    global _cache
    if _cache:
        return _cache

    missing = []
    files = {
        'yield_model':    'yield_model.pkl',
        'crop_model':     'crop_model.pkl',
        'scaler':         'scaler.pkl',
        'encoder':        'encoder.pkl',
    }
    for key, fname in files.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            _cache[key] = joblib.load(path)
        else:
            missing.append(fname)

    if missing:
        print(err(f"Missing model files: {missing}"))
        print(warn("Run  python3 phase2_pipeline.py  first to train models."))
        sys.exit(1)

    return _cache

# ─────────────────────── Input helpers ──────────────────────────────────────
VALID_SOIL_TYPES   = ['Loamy', 'Red', 'Black', 'Alluvial', 'Sandy',
                      'Laterite', 'Clay', 'Silt']
VALID_IRRIGATION   = ['Canal', 'Rainfed', 'Tube Well', 'Drip', 'Sprinkler']
VALID_SEASONS      = ['Kharif', 'Rabi', 'Zaid', 'Annual']
VALID_DISTRICTS    = []   # open-ended; any string accepted
VALID_STATES       = []   # open-ended

def _prompt(label, cast=float, choices=None, default=None):
    """Prompt user for a value with optional casting and validation."""
    while True:
        hint = ""
        if choices:
            hint = f"  [{' / '.join(choices)}]"
        if default is not None:
            hint += f"  (default: {default})"
        val = input(f"  {CYAN}{label}{hint}: {RESET}").strip()
        if val == '' and default is not None:
            return default
        if choices:
            # case-insensitive match
            match = next((c for c in choices if c.lower() == val.lower()), None)
            if match:
                return match
            print(warn(f"  Choose from: {choices}"))
            continue
        try:
            return cast(val) if val else default
        except ValueError:
            print(warn(f"  Expected {cast.__name__}. Try again."))


def gather_inputs():
    print("\n" + h("─" * 60))
    print(h("  Enter Farming Conditions", YELLOW))
    print(h("─" * 60))

    data = {}

    # Categorical
    data['District']   = _prompt("District", cast=str, default='Angul')
    data['State']      = _prompt("State",    cast=str, default='Odisha')
    data['Season']     = _prompt("Season",   cast=str, choices=VALID_SEASONS, default='Kharif')
    data['Soil_Type']  = _prompt("Soil Type",cast=str, choices=VALID_SOIL_TYPES, default='Loamy')
    data['Irrigation'] = _prompt("Irrigation",cast=str,choices=VALID_IRRIGATION,  default='Canal')

    print()
    # Numeric
    data['Rainfall']            = _prompt("Rainfall (mm)",          float,  default=1200.0)
    data['Temperature']         = _prompt("Temperature (°C)",       float,  default=30.0)
    data['Nitrogen']            = _prompt("Nitrogen (kg/ha)",       float,  default=300.0)
    data['Phosphorus']          = _prompt("Phosphorus (kg/ha)",     float,  default=200.0)
    data['Potassium']           = _prompt("Potassium (kg/ha)",      float,  default=250.0)
    data['Humidity']            = _prompt("Humidity (%)",           float,  default=70.0)
    data['Soil_pH']             = _prompt("Soil pH",                float,  default=6.5)
    data['Area']                = _prompt("Area (hectares)",        float,  default=10000.0)
    data['Fertilizer_kg_per_ha']= _prompt("Fertilizer (kg/ha)",    float,  default=80.0)
    data['Previous_Yield']      = _prompt("Previous Yield (t/ha)", float,  default=1.5)
    data['Year']                = _prompt("Year",                   int,    default=2025)

    # Derived / default values
    data['Yield_Change']         = 0.0
    data['Rainfall_Score']       = 2
    data['Weather_Score']        = 2
    data['Soil_Fertility_Index'] = 0.55
    data['WSI_raw']              = 0.03
    data['Humidity_Factor']      = 1.0
    data['Irrigation_Factor']    = 0.95
    data['Season_Factor']        = 1.0
    data['Temp_Factor']          = 1.0
    data['Rainfall_Factor']      = 1.0
    data['Water_Stress_Index']   = 0.03
    data['Rainfall_Distribution']= 'Moderate'
    data['Water_Stress_Level']   = 'Medium Stress'
    data['Crop_Category']        = 'Cereal'
    data['Weather_Forecast']     = 'Balanced'
    data['Production']           = 0.0
    data['Yield']                = data['Previous_Yield']

    return data


def build_features(row: dict, arts: dict, task: str) -> np.ndarray:
    scaler_dict = arts['scaler']
    le_map      = arts['encoder']

    scaler      = scaler_dict[f'{task}_scaler']
    feat_list   = scaler_dict[f'{task}_features']

    r = dict(row)   # copy

    # --- Ordinal encodings ---
    ordinal = {
        'Rainfall_Distribution': {'Low':0,'Moderate':1,'Good':2,'Heavy':3},
        'Water_Stress_Level':    {'Low Stress':0,'Medium Stress':1,'High Stress':2},
    }
    for col, mp in ordinal.items():
        r[col + '_encoded'] = mp.get(r.get(col,'Moderate'), 1)

    # --- Label encodings ---
    for col in ['District','State','Soil_Type','Irrigation','Season',
                'Crop_Category','Weather_Forecast']:
        le = le_map.get(col)
        val = str(r.get(col, ''))
        if le is not None and val in le.classes_:
            r[col + '_encoded'] = int(le.transform([val])[0])
        else:
            r[col + '_encoded'] = 0

    # --- Derived season/crop encodings ---
    r['Season_Factor']    = {'Kharif':0.9,'Rabi':1.0,'Zaid':1.15,'Annual':1.0}.get(r.get('Season','Kharif'),1.0)

    # --- Original engineered features ---
    N, P, K = r.get('Nitrogen',200), r.get('Phosphorus',150), r.get('Potassium',200)
    r['Nutrient_Ratio']        = (N+P+K)/3
    r['Climate_Index']         = (r.get('Rainfall',1000)+r.get('Temperature',30)+r.get('Humidity',70))/3
    r['Soil_Health']           = (r.get('Soil_pH',6.5)+r.get('Soil_Fertility_Index',0.55))/2
    r['Water_Index']           = (r.get('Water_Stress_Index',0.03)+r.get('Irrigation_Factor',0.95))/2
    r['Yield_Trend']           = r.get('Previous_Yield',1.0)+r.get('Yield_Change',0.0)
    fert = r.get('Fertilizer_kg_per_ha',50)
    r['Fertilizer_Efficiency'] = r.get('Yield',1.0)/fert if fert > 0 else 0

    # --- Phase 2 new features ---
    r['NPK_Ratio']                  = N / (P + K + 1.0)
    r['Rainfall_Temperature_Index'] = r.get('Rainfall',1000) * r.get('Temperature',30)
    r['Soil_Moisture_Index']        = r.get('Humidity',70) * r.get('Rainfall',1000)
    r['Fertilizer_Interaction']     = fert * r.get('Soil_Fertility_Index',0.55)
    r['Climate_Risk_Index']         = r.get('Weather_Score',2) * r.get('Rainfall_Score',2)
    r['Water_Availability_Index']   = r.get('Irrigation_Factor',0.95) * r.get('Rainfall_Factor',1.0)
    r['Crop_Season_Interaction']    = r.get('Crop_Category_encoded',0) * r.get('Season_encoded',0)
    r['District_Season_Yield_Avg']  = r.get('Yield', r.get('Previous_Yield', 1.5))

    # --- Build vector ---
    vec = [float(r.get(f, 0.0)) for f in feat_list]
    X   = np.array([vec])
    return scaler.transform(X)


# ─────────────────────── Predictions ────────────────────────────────────────
def predict_yield(row, arts):
    X    = build_features(row, arts, 'yield')
    pred = float(arts['yield_model'].predict(X)[0])
    return pred


def predict_crop(row, arts):
    X    = build_features(row, arts, 'crop')
    pred = int(arts['crop_model'].predict(X)[0])
    le   = arts['encoder']['Crops']
    name = le.inverse_transform([pred])[0]

    probs = {}
    if hasattr(arts['crop_model'], 'predict_proba'):
        p_arr = arts['crop_model'].predict_proba(X)[0]
        classes = le.classes_
        probs = {classes[i]: round(float(p_arr[i])*100, 2)
                 for i in np.argsort(p_arr)[::-1][:5]}
    return name, probs


def confidence_band(r2_proxy):
    """Map prediction confidence from model R²."""
    if r2_proxy >= 0.98: return "🔵 Very High (>98%)"
    if r2_proxy >= 0.95: return "🟢 High (95-98%)"
    if r2_proxy >= 0.90: return "🟡 Moderate (90-95%)"
    return "🔴 Low (<90%)"


# ─────────────────────── Display results ────────────────────────────────────
def display_results(row, yield_val, crop_name, probs):
    print("\n" + h("═" * 60, GREEN))
    print(h("  🌾  PREDICTION RESULTS", GREEN))
    print(h("═" * 60, GREEN))

    print(f"\n  {h('Input Summary:', YELLOW)}")
    show = ['District','State','Season','Rainfall','Temperature',
            'Soil_Type','Irrigation','Nitrogen','Phosphorus',
            'Potassium','Humidity','Soil_pH']
    for k in show:
        if k in row:
            print(f"    {k:30s}: {row[k]}")

    print(f"\n  {h('🌱  Predicted Crop Yield:', GREEN)}")
    print(f"    {BOLD}{yield_val:.4f} tonnes / hectare{RESET}")
    print(f"    Confidence: {confidence_band(yield_val)}")

    print(f"\n  {h('🌿  Recommended Crop:', GREEN)}")
    print(f"    {BOLD}{crop_name}{RESET}")

    if probs:
        print(f"\n  {h('  Top-5 Crop Probabilities:', CYAN)}")
        for crop, pct in probs.items():
            bar = '█' * int(pct / 5)
            print(f"    {crop:20s} {bar:20s} {pct:.1f}%")

    print(h("═" * 60, GREEN))


# ─────────────────────── Phase 6 — Interactive CLI Loop ─────────────────────
def main():
    print("\n" + h("╔" + "═"*58 + "╗", CYAN))
    print(h("║   🌾  CROP ML SYSTEM — Manual Testing Interface         ║", CYAN))
    print(h("╚" + "═"*58 + "╝", CYAN))
    print(f"\n  Loading trained models from: {MODELS_DIR}")

    arts = _load_all()

    # List available model files
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
    print(ok(f"Loaded {len(model_files)} model artefacts: {model_files}"))

    run_count = 0
    while True:
        run_count += 1
        print(f"\n{h(f'─── Prediction #{run_count} ───', YELLOW)}")

        try:
            row = gather_inputs()

            print(f"\n  {CYAN}⏳ Running predictions...{RESET}")
            yield_val           = predict_yield(row, arts)
            crop_name, probs    = predict_crop(row, arts)
            display_results(row, yield_val, crop_name, probs)

        except KeyboardInterrupt:
            print(f"\n{warn('Interrupted.')}")
            break
        except Exception as e:
            print(err(f"Prediction error: {e}"))
            import traceback; traceback.print_exc()

        # Phase 6 — loop
        print()
        again = input(f"  {YELLOW}Do you want to predict again? (yes/no): {RESET}").strip().lower()
        if again not in ('yes', 'y'):
            break

    print(f"\n{ok(f'Session ended after {run_count} prediction(s). Goodbye! 🌾')}\n")


if __name__ == '__main__':
    main()
