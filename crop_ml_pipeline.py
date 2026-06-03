import time 
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import json, warnings, pickle, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split, cross_val_score,
    StratifiedKFold, KFold, learning_curve, RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report)
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, VotingRegressor, VotingClassifier,
    StackingRegressor, StackingClassifier, IsolationForest,
    ExtraTreesRegressor, ExtraTreesClassifier)
from sklearn.feature_selection import (mutual_info_regression, mutual_info_classif,
    RFE)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.inspection import PartialDependenceDisplay

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
import shap
import torch
import torch.nn as nn

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), 'smart_farming_final_dataset.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================================
# STEP 1: DATA LOADING & PREPROCESSING
# ============================================================================
def load_data():
    print("=" * 70)
    print("STEP 1: Loading & Preprocessing Data")
    print("=" * 70)
    df = pd.read_csv(DATA_PATH)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df


def handle_missing_values(df):
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    for c in num_cols:
        if df[c].isnull().sum() > 0:
            df[c].fillna(df[c].median(), inplace=True)
    for c in cat_cols:
        if df[c].isnull().sum() > 0:
            df[c].fillna(df[c].mode()[0], inplace=True)
    print(f"  Missing values after handling: {df.isnull().sum().sum()}")
    return df


def remove_outliers_iqr(df, cols, factor=1.5):
    df = df.copy()
    initial = len(df)
    for c in cols:
        if c in df.columns and df[c].dtype in [np.float64, np.int64, np.float32]:
            Q1, Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
            IQR = Q3 - Q1
            mask = (df[c] >= Q1 - factor * IQR) & (df[c] <= Q3 + factor * IQR)
            df = df[mask]
    print(f"  Outlier removal (IQR): {initial} -> {len(df)} rows ({initial - len(df)} removed)")
    return df


def encode_features(df):
    df = df.copy()
    # Ordinal encoding
    ordinal_maps = {
        'Rainfall_Distribution': {'Low': 0, 'Moderate': 1, 'Good': 2, 'Heavy': 3},
        'Water_Stress_Level': {'Low Stress': 0, 'Medium Stress': 1, 'High Stress': 2},
    }
    for col, mapping in ordinal_maps.items():
        if col in df.columns:
            df[col + '_encoded'] = df[col].map(mapping).fillna(0).astype(int)

    # Label encode remaining categoricals for tree models
    label_encoders = {}
    cat_cols = ['District', 'State', 'Soil_Type', 'Irrigation', 'Season',
                'Crop_Category', 'Weather_Forecast', 'Crop_Season_Interaction']
    for c in cat_cols:
        if c in df.columns:
            le = LabelEncoder()
            df[c + '_encoded'] = le.fit_transform(df[c].astype(str))
            label_encoders[c] = le

    # Encode target for classification
    if 'Crops' in df.columns:
        le_crops = LabelEncoder()
        df['Crops_encoded'] = le_crops.fit_transform(df['Crops'])
        label_encoders['Crops'] = le_crops

    print(f"  Encoded {len(label_encoders)} categorical features")
    return df, label_encoders


# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================
def engineer_features(df):
    print("\n" + "=" * 70)
    print("STEP 2: Feature Engineering")
    print("=" * 70)
    df = df.copy()

    df['Nutrient_Ratio'] = (df['Nitrogen'] + df['Phosphorus'] + df['Potassium']) / 3
    df['Climate_Index'] = (df['Rainfall'] + df['Temperature'] + df['Humidity']) / 3
    df['Soil_Health'] = (df['Soil_pH'] + df['Soil_Fertility_Index']) / 2
    df['Water_Index'] = (df['Water_Stress_Index'] + df['Irrigation_Factor']) / 2
   
    # Phase 2 - Advanced Feature Engineering
    df['NPK_Ratio'] = df['Nitrogen'] / (df['Phosphorus'] + df['Potassium'] + 1)
    df['Rainfall_Temperature_Index'] = df['Rainfall'] * df['Temperature']
    df['Soil_Moisture_Index'] = df['Humidity'] * df['Rainfall']
    if 'Crops' in df.columns and 'Season' in df.columns:
        df['Crop_Season_Interaction'] = df['Crops'].astype(str) + "_" + df['Season'].astype(str)
    
    df['Fertilizer_Interaction'] = df['Fertilizer_kg_per_ha'] * df['Soil_Fertility_Index']
    df['Climate_Risk_Index'] = df['Weather_Score'] * df['Rainfall_Score']
    df['Water_Availability_Index'] = df['Irrigation_Factor'] * df['Rainfall_Factor']

    new_feats = [
    'Nutrient_Ratio',
    'Climate_Index',
    'Soil_Health',
    'Water_Index',
    
    'NPK_Ratio',
    'Rainfall_Temperature_Index',
    'Soil_Moisture_Index',
    'Fertilizer_Interaction',
    'Climate_Risk_Index',
    'Water_Availability_Index'
]
    if 'Crop_Season_Interaction' in df.columns:
        new_feats.append('Crop_Season_Interaction')

    print(f"  Created {len(new_feats)} new features: {new_feats}")
    return df


# ============================================================================
# STEP 3: FEATURE SELECTION
# ============================================================================
def select_features(X, y, task='regression'):
    print("\n" + "=" * 70)
    print(f"STEP 3: Feature Selection ({task})")
    print("=" * 70)

    # 1. Correlation-based removal
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = [c for c in upper.columns if any(upper[c] > 0.95)]
    print(f"  High correlation features (>0.95): {high_corr}")

    # 2. Mutual Information
    if task == 'regression':
        mi = mutual_info_regression(X, y, random_state=42)
    else:
        mi = mutual_info_classif(X, y, random_state=42)
    mi_scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    mi_top = mi_scores.head(20).index.tolist()
    print(f"  Top 10 MI features: {mi_scores.head(10).index.tolist()}")

    # 3. Random Forest importance
    if task == 'regression':
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    else:
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    rf_top = imp.head(20).index.tolist()
    print(f"  Top 10 RF features: {imp.head(10).index.tolist()}")

    # 4. RFE
    rfe = RFE(rf, n_features_to_select=min(20, X.shape[1]), step=3)
    rfe.fit(X, y)
    rfe_features = X.columns[rfe.support_].tolist()
    print(f"  RFE selected: {len(rfe_features)} features")

    # Union of top features from all methods
    selected = list(set(mi_top) | set(rf_top) | set(rfe_features))
    # Remove high-corr duplicates
    selected = [f for f in selected if f not in high_corr]
    if len(selected) == 0:
        selected = mi_top[:15]
    print(f"  Final selected features: {len(selected)}")

    # Save feature importance plot
    fig, ax = plt.subplots(figsize=(12, 8))
    imp.head(20).plot(kind='barh', ax=ax, color='#2196F3')
    ax.set_title(f'Feature Importance ({task.title()})', fontsize=14)
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'feature_importance_{task}.png'), dpi=150)
    plt.close()

    return selected, imp


# ============================================================================
# STEP 4 & 5: MODEL BUILDING + OPTUNA TUNING
# ============================================================================
class PyTorchMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden // 2, output_dim)
        )
    def forward(self, x):
        return self.net(x)


def train_pytorch_model(X_train, y_train, X_test, y_test, task='regression', epochs=150):
    device = torch.device('cpu')
    X_tr = torch.FloatTensor(X_train.values if hasattr(X_train, 'values') else X_train).to(device)
    X_te = torch.FloatTensor(X_test.values if hasattr(X_test, 'values') else X_test).to(device)

    if task == 'regression':
        y_tr = torch.FloatTensor(y_train.values if hasattr(y_train, 'values') else y_train).unsqueeze(1).to(device)
        output_dim = 1
        criterion = nn.MSELoss()
    else:
        n_classes = len(np.unique(y_train))
        y_tr = torch.LongTensor(y_train.values if hasattr(y_train, 'values') else y_train).to(device)
        output_dim = n_classes
        criterion = nn.CrossEntropyLoss()

    model = PyTorchMLP(X_tr.shape[1], output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X_tr)
        loss = criterion(out, y_tr)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

    model.eval()
    with torch.no_grad():
        preds = model(X_te)
        if task == 'regression':
            preds = preds.squeeze().numpy()
        else:
            preds = preds.argmax(dim=1).numpy()
    return preds, model


def build_regression_models():
    return {
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=15,
            min_samples_split=5, random_state=42, n_jobs=1),
        'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0),
        'LightGBM': lgb.LGBMRegressor(n_estimators=200, max_depth=8, learning_rate=0.1,
            num_leaves=31, random_state=42, verbose=-1),
        'CatBoost': cb.CatBoostRegressor(iterations=200, depth=6, learning_rate=0.1,
            random_seed=42, verbose=0),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200,max_depth=3,
            learning_rate=0.1, random_state=42),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=200, max_depth=15,
            min_samples_split=5, random_state=42, n_jobs=1),
    }


def build_classification_models(n_classes):
    return {
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=15,
            min_samples_split=5, random_state=42, n_jobs=1),
        'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, verbosity=0, eval_metric='mlogloss'),
        'LightGBM': lgb.LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
            num_leaves=31, random_state=42, verbose=-1),
        'CatBoost': cb.CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1,
            random_seed=42, verbose=0),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=200, max_depth=15,
            min_samples_split=5, random_state=42, n_jobs=1),
    }


def optuna_tune_xgb(X_train, y_train, task='regression', n_trials=30):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42, 'verbosity': 0,
        }
        if task == 'regression':
            model = xgb.XGBRegressor(**params)
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2', n_jobs=1)
        else:
            params['eval_metric'] = 'mlogloss'
            model = xgb.XGBClassifier(**params)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=1)
        return scores.mean()

    try:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        print(f"  Optuna XGBoost best: {study.best_value:.4f}")
        return study.best_params
    except Exception as e:
        print(f"  Optuna XGBoost failed ({e}). Using Default params.")
        return {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1}


def optuna_tune_lgbm(X_train, y_train, task='regression', n_trials=30):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42, 'verbose': -1,
        }
        if task == 'regression':
            model = lgb.LGBMRegressor(**params)
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2', n_jobs=1)
        else:
            model = lgb.LGBMClassifier(**params)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=1)
        return scores.mean()

    try:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        print(f"  Optuna LightGBM best: {study.best_value:.4f}")
        return study.best_params
    except Exception as e:
        print(f"  Optuna LightGBM failed ({e}). Fallback to Default params.")
        return {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.1, 'num_leaves': 31}


def optuna_tune_catboost(X_train, y_train, task='regression', n_trials=20):
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'random_seed': 42, 'verbose': 0,
        }
        if task == 'regression':
            model = cb.CatBoostRegressor(**params)
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2', n_jobs=1)
        else:
            model = cb.CatBoostClassifier(**params)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=1)
        return scores.mean()

    try:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        print(f"  Optuna CatBoost best: {study.best_value:.4f}")
        return study.best_params
    except Exception as e:
        print(f"  Optuna CatBoost failed ({e}). Fallback to Random/Default params.")
        return {'iterations': 200, 'depth': 6, 'learning_rate': 0.1, 'l2_leaf_reg': 3.0}


# ============================================================================
# STEP 6: CROSS VALIDATION
# ============================================================================
def cross_validate_model(model, X, y, task='regression'):
    if task == 'regression':
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        r2 = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=1)
        rmse = -cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=1)
        mae = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=1)
        return {'R2_mean': r2.mean(), 'R2_std': r2.std(),
                'RMSE_mean': rmse.mean(), 'MAE_mean': mae.mean()}
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=1)
        f1 = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted', n_jobs=1)
        return {'Acc_mean': acc.mean(), 'Acc_std': acc.std(),
                'F1_mean': f1.mean(), 'F1_std': f1.std()}


def evaluate_model_performance(model, X_train, y_train, X_test, y_test, name, task='regression'):
    os.makedirs(os.path.join(RESULTS_DIR, 'testing'), exist_ok=True)
    tests = {}

    train_sizes, train_scores, test_scores = learning_curve(
    model,
    pd.concat([X_train, X_test]),
    pd.concat([y_train, y_test]),
    cv=cv,
    scoring=scoring
)
    
    # 1. Overfitting Detection
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    if task == 'regression':
        train_score = r2_score(y_train, train_preds)
        test_score = r2_score(y_test, test_preds)
    else:
        train_score = f1_score(y_train, train_preds, average='weighted', zero_division=0)
        test_score = f1_score(y_test, test_preds, average='weighted', zero_division=0)
    
    overfit_gap = train_score - test_score
    tests['Train_Score'] = train_score
    tests['Test_Score'] = test_score
    tests['Overfit_Gap'] = overfit_gap
    tests['Overfitting'] = bool(overfit_gap > 0.15)
    
    # 2. Learning Curve
    cv = KFold(n_splits=5, shuffle=True, random_state=42) if task == 'regression' else StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = 'r2' if task == 'regression' else 'f1_weighted'
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            model, pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), 
            cv=cv, scoring=scoring, n_jobs=1, train_sizes=np.linspace(0.1, 1.0, 5))
        
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)

        print("Learning Curve Train Scores:", train_mean)
        print("Learning Curve Validation Scores:", test_mean)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
        plt.title(f"Learning Curve - {name}")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.legend(loc="best")
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, 'testing', f'{name}_learning_curve.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f"  Learning Curve failed for {name}: {e}")
        
    with open(os.path.join(RESULTS_DIR, 'testing', f'{name}_validation.json'), 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.float32, np.float64, float)) else v for k, v in tests.items()}, f, indent=2)
    
    return tests

# ============================================================================
# STEP 7: EXPLAINABILITY
# ============================================================================
def explain_model(model, X, feature_names, task='regression'):
    print("\n" + "=" * 70)
    print(f"STEP 7: Model Explainability ({task})")
    print("=" * 70)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X[:200])
        fig, ax = plt.subplots(figsize=(12, 8))
        if task == 'classification' and isinstance(shap_values, list):
            shap.summary_plot(shap_values[0], X[:200], feature_names=feature_names, show=False)
        else:
            shap.summary_plot(shap_values, X[:200], feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'shap_summary_{task}.png'), dpi=150, bbox_inches='tight')
        plt.close('all')
        print("  SHAP summary plot saved")
    except Exception as e:
        print(f"  SHAP error: {e}")

    # Partial Dependence Plot
    try:
        if hasattr(model, 'feature_importances_'):
            top_feats = np.argsort(model.feature_importances_)[-4:]
            fig, ax = plt.subplots(figsize=(14, 8))
            PartialDependenceDisplay.from_estimator(model, X[:300], top_feats, ax=ax)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f'pdp_{task}.png'), dpi=150, bbox_inches='tight')
            plt.close('all')
            print("  PDP saved")
    except Exception as e:
        print(f"  PDP error: {e}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def prepare_features(df):
    exclude = [
        'Yield',
        'Production',
        'Crops',
        'Crops_encoded',

        'Previous_Yield',
        'Yield_Change',

        'Fertilizer_Efficiency',
        'Yield_Efficiency',

         'Crop_Season_Interaction',
        'Crop_Season_Interaction_encoded',

        'District',
        'State',
        'Soil_Type',
        'Irrigation',
        'Season',
        'Crop_Category',
        'Weather_Forecast',
        'Rainfall_Distribution',
        'Water_Stress_Level'
    ]
    feature_cols = [c for c in df.columns if c not in exclude
                    and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
    return feature_cols


def run_yield_prediction(df, feature_cols):
    print("\n" + "=" * 70)
    print("STEP 4a: YIELD PREDICTION (Regression)")
    print("=" * 70)

    X = df[feature_cols].copy()
    y = df['Yield'].copy()

    # Feature selection
    selected, imp = select_features(X, y, task='regression')
    X = X[selected]

    



    
    # Scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)
    print(f"\n  Train: {X_train.shape}, Test: {X_test.shape}")

    # Train base models
    models = build_regression_models()
    results = {}

    for name, model in models.items():
        start_time = time.time()

        model.fit(X_train, y_train)

        training_time = time.time() - start_time
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        cv = cross_validate_model(model, X_scaled, y, 'regression')
        results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'CV_R2': cv['R2_mean']}
        print(f"  {name:20s} | R²={r2:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f} | CV_R²={cv['R2_mean']:.4f}")

    # Neural Net
    nn_preds, nn_model = train_pytorch_model(X_train, y_train, X_test, y_test, 'regression')
    r2 = r2_score(y_test, nn_preds)
    rmse = np.sqrt(mean_squared_error(y_test, nn_preds))
    mae = mean_absolute_error(y_test, nn_preds)
    results['NeuralNet'] = {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'CV_R2': r2}
    print(f"  {'NeuralNet':20s} | R²={r2:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f}")

    # Optuna tuning on top 3
    print("\n  --- Optuna Hyperparameter Tuning ---")
    xgb_params = optuna_tune_xgb(X_train, y_train, 'regression', n_trials=30)
    lgbm_params = optuna_tune_lgbm(X_train, y_train, 'regression', n_trials=30)
    cb_params = optuna_tune_catboost(X_train, y_train, 'regression', n_trials=20)

    # Retrain with best params
    xgb_tuned = xgb.XGBRegressor(**xgb_params, random_state=42, verbosity=0)
    lgbm_tuned = lgb.LGBMRegressor(**lgbm_params, random_state=42, verbose=-1)
    cb_tuned = cb.CatBoostRegressor(**cb_params, random_seed=42, verbose=0)

    for name, model in [('XGB_tuned', xgb_tuned), ('LGBM_tuned', lgbm_tuned), ('CB_tuned', cb_tuned)]:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        results[name] = {'R2': r2,'Training_Time': training_time, 'RMSE': rmse, 'MAE': mean_absolute_error(y_test, preds)}
        print(f"  {name:20s} | R²={r2:.4f} | RMSE={rmse:.4f}")
        models[name] = model

    # Ensembles
    print("\n  --- Ensembles ---")
    voting = VotingRegressor([('xgb', xgb_tuned), ('lgbm', lgbm_tuned), ('cb', cb_tuned)])
    
    weights = [results['XGB_tuned']['R2'], results['LGBM_tuned']['R2'], results['CB_tuned']['R2']]
    weighted_voting = VotingRegressor([('xgb', xgb_tuned), ('lgbm', lgbm_tuned), ('cb', cb_tuned)], weights=weights)

    stacking = StackingRegressor(
        estimators=[('xgb', xgb_tuned), ('lgbm', lgbm_tuned), ('cb', cb_tuned)],
        final_estimator=Ridge(alpha=1.0))
        
    blended = StackingRegressor(
        estimators=[('xgb', xgb_tuned), ('lgbm', lgbm_tuned), ('cb', cb_tuned)],
        final_estimator=RandomForestRegressor(n_estimators=50, random_state=42), passthrough=True)

    ensemble_models = {'Voting': voting, 'Weighted_Voting': weighted_voting, 'Stacking': stacking, 'Blended': blended}
    
    for ename, emodel in ensemble_models.items():
        emodel.fit(X_train, y_train)
        preds = emodel.predict(X_test)
        r2 = r2_score(y_test, preds)
        results[ename] = {'R2': r2, 'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
                          'MAE': mean_absolute_error(y_test, preds)}
        print(f"  {ename:20s} | R²={r2:.4f}")
        models[ename] = emodel

    # Select best
    best_name = max(results, key=lambda k: results[k]['R2'])
    print(f"\n  *** BEST YIELD MODEL: {best_name} (R²={results[best_name]['R2']:.4f}) ***")
    
    best_model = models.get(best_name)
    if best_model is None:
        best_model = ensemble_models.get(best_name, stacking)

    # Explainability
    explainable = models.get('XGB_tuned', models.get('XGBoost'))
    explain_model(explainable, X_test, selected, 'regression')

    # Evaluate performance
    evaluate_model_performance(best_model, X_train, y_train, X_test, y_test, f"Yield_{best_name}", 'regression')
    
    # Save
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'yield_model.pkl'))
    # We will combine scalers and features in main()
    with open(os.path.join(RESULTS_DIR, 'yield_metrics.json'), 'w') as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)

    # Save runtime analysis table
    runtime_table = pd.DataFrame([
                {
                    'Model': k,
                    'Training_Time_sec': v['Training_Time']
                }
                for k, v in results.items()
                if 'Training_Time' in v
    ])

    runtime_table.to_csv(
    os.path.join(RESULTS_DIR, 'yield_runtime_analysis.csv'),
    index=False
)   
    print("  Models and metrics saved!")

    return best_model, scaler, selected, results
     

    #Ablation Study Section

def run_ablation_study(df):
    print("\n" + "="*70)
    print("ABLATION STUDY")
    print("="*70)

    experiments = {
        "Soil": [
            'Nitrogen', 'Phosphorus', 'Potassium',
            'Soil_pH', 'Soil_Fertility_Index'
        ],

        "Climate": [
            'Rainfall', 'Temperature',
            'Humidity', 'Weather_Score'
        ],

        "Soil+Climate": [
            'Nitrogen', 'Phosphorus', 'Potassium',
            'Soil_pH', 'Soil_Fertility_Index',
            'Rainfall', 'Temperature',
            'Humidity', 'Weather_Score'
        ],

        "Full Model": prepare_features(df)
    }

    results = []

    y = df['Crops_encoded']

    for name, features in experiments.items():

        available_features = [f for f in features if f in df.columns]

        X = df[available_features]

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(
            y_test,
            preds,
            average='weighted'
        )

        results.append({
            'Features': name,
            'Accuracy': acc,
            'F1': f1
        })

        print(
            f"{name:15s} | "
            f"Accuracy={acc:.4f} | "
            f"F1={f1:.4f}"
        )

    ablation_df = pd.DataFrame(results)

    ablation_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            'ablation_study.csv'
        ),
        index=False
    )

    print("\nAblation study saved.")

    return ablation_df



def run_crop_recommendation(df, feature_cols):
    print("\n" + "=" * 70)
    print("STEP 4b: CROP RECOMMENDATION (Classification)")
    print("=" * 70)

    X = df[feature_cols].copy()
    y = df['Crops_encoded'].copy()
    crop_names = df['Crops'].unique()
    n_classes = len(crop_names)
    print(f"  Classes: {n_classes} crops -> {list(crop_names)}")

    # Feature selection
    selected, imp = select_features(X, y, task='classification')
    X = X[selected]

    # Scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\n  Train: {X_train.shape}, Test: {X_test.shape}")

    # Train base models
    models = build_classification_models(n_classes)
    results = {}

    for name, model in models.items():

        start_time = time.time()

        model.fit(X_train, y_train)

        training_time = time.time() - start_time
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='weighted', zero_division=0)
        rec = recall_score(y_test, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
        cv = cross_validate_model(model, X_scaled, y, 'classification')
        results[name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'CV_Acc_Mean': cv['Acc_mean'],
        'CV_Acc_Std': cv['Acc_std'],
        'CV_F1_Mean': cv['F1_mean'],
        'CV_F1_Std': cv['F1_std'],
        'Training_Time': training_time
    }

        print(
            f"{name:20s} | "
            f"Acc={acc:.4f} | "
            f"F1={f1:.4f} | "
            f"CV_Acc={cv['Acc_mean']:.4f} ± {cv['Acc_std']:.4f}"
        )

    # Neural Net
    nn_preds, nn_model = train_pytorch_model(X_train, y_train, X_test, y_test, 'classification')
    acc = accuracy_score(y_test, nn_preds)
    f1 = f1_score(y_test, nn_preds, average='weighted', zero_division=0)
    results['NeuralNet'] = {'Accuracy': acc, 'F1': f1, 'Training_Time': training_time}
    print(f"  {'NeuralNet':20s} | Acc={acc:.4f} | F1={f1:.4f}")

    # Optuna tuning
    print("\n  --- Optuna Hyperparameter Tuning ---")
    xgb_params = optuna_tune_xgb(X_train, y_train, 'classification', n_trials=30)
    lgbm_params = optuna_tune_lgbm(X_train, y_train, 'classification', n_trials=30)
    cb_params = optuna_tune_catboost(X_train, y_train, 'classification', n_trials=20)

    xgb_params['eval_metric'] = 'mlogloss'
    xgb_tuned = xgb.XGBClassifier(**xgb_params, random_state=42, verbosity=0)
    lgbm_tuned = lgb.LGBMClassifier(**lgbm_params, random_state=42, verbose=-1)
    cb_tuned = cb.CatBoostClassifier(**cb_params, random_seed=42, verbose=0)

    for name, model in [('XGB_tuned', xgb_tuned), ('LGBM_tuned', lgbm_tuned), ('CB_tuned', cb_tuned)]:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
        results[name] = {'Accuracy': acc, 'F1': f1}
        print(f"  {name:20s} | Acc={acc:.4f} | F1={f1:.4f}")
        models[name] = model

    # Ensembles
    print("\n  --- Ensembles ---")
    voting = VotingClassifier(
        estimators=[('xgb', xgb_tuned), ('lgbm', lgbm_tuned), ('cb', cb_tuned)],
        voting='soft')
        
    weights = [results['XGB_tuned']['F1'], results['LGBM_tuned']['F1'], results['CB_tuned']['F1']]
    weighted_voting = VotingClassifier(
        estimators=[('xgb', xgb_tuned), ('lgbm', lgbm_tuned), ('cb', cb_tuned)],
        voting='soft', weights=weights)

    stacking = StackingClassifier(
        estimators=[('xgb', xgb_tuned), ('lgbm', lgbm_tuned), ('cb', cb_tuned)],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42))
        
    blended = StackingClassifier(
        estimators=[('xgb', xgb_tuned), ('lgbm', lgbm_tuned), ('cb', cb_tuned)],
        final_estimator=RandomForestClassifier(n_estimators=50, random_state=42), passthrough=True)

    ensemble_models = {'Voting': voting, 'Weighted_Voting': weighted_voting, 'Stacking': stacking, 'Blended': blended}
    
    for ename, emodel in ensemble_models.items():
        emodel.fit(X_train, y_train)
        preds = emodel.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
        results[ename] = {'Accuracy': acc, 'F1': f1}
        print(f"  {ename:20s} | Acc={acc:.4f} | F1={f1:.4f}")
        models[ename] = emodel

    # Best model
    best_name = max(results, key=lambda k: results[k].get('F1', 0))
    print(f"\n  *** BEST CROP MODEL: {best_name} (F1={results[best_name]['F1']:.4f}) ***")
    
    best_model = models.get(best_name)
    if best_model is None:
        best_model = ensemble_models.get(best_name, stacking)

    # Confusion matrix
    final_preds = best_model.predict(X_test) if hasattr(best_model, 'predict') else preds
    cm = confusion_matrix(y_test, final_preds)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix — Crop Recommendation')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # Classification report
    print("\n  Classification Report:")
    print(classification_report(y_test, final_preds, zero_division=0))

    # Explainability
    explainable = models.get('XGB_tuned', models.get('XGBoost'))
    explain_model(explainable, X_test, selected, 'classification')

    # Evaluate performance
    evaluate_model_performance(best_model, X_train, y_train, X_test, y_test, f"Crop_{best_name}", 'classification')

    # Save
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'crop_model.pkl'))
    with open(os.path.join(RESULTS_DIR, 'crop_metrics.json'), 'w') as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)

    # Save runtime analysis table
    runtime_table = pd.DataFrame([
                {
                    'Model': k,
                    'Training_Time_sec': v['Training_Time']
                }
                for k, v in results.items()
                if 'Training_Time' in v
    ])

    runtime_table.to_csv(
    os.path.join(RESULTS_DIR, 'crop_runtime_analysis.csv'),
    index=False
)   
    print("  Models and metrics saved!")

    return best_model, scaler, selected, results


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("=" * 70)
    print("  CROP ML PIPELINE — ULTRA HIGH PERFORMANCE SYSTEM")
    print("=" * 70)

    # Step 1: Load & preprocess
    df = load_data()
    df = handle_missing_values(df)

    # Step 2: Feature Engineering (Do this BEFORE encoding so string features can be encoded)
    df = engineer_features(df)

    # Step 1b: Encode
    df, label_encoders = encode_features(df)

    # Step 1c: Outlier removal on key numeric columns
    outlier_cols = ['Yield', 'Rainfall', 'Temperature', 'Fertilizer_kg_per_ha',
                    'Nitrogen', 'Phosphorus', 'Potassium', 'Humidity', 'Area']
    df = remove_outliers_iqr(df, outlier_cols)
    df = df.reset_index(drop=True)

    # Re-encode Crops after outlier removal to ensure contiguous labels
    le_crops = LabelEncoder()
    df['Crops_encoded'] = le_crops.fit_transform(df['Crops'])
    label_encoders['Crops'] = le_crops

    # Prepare feature columns
    feature_cols = prepare_features(df)
    print(f"\n  Feature columns ({len(feature_cols)}): {feature_cols}")

    # Save encoders as encoder.pkl
    joblib.dump(label_encoders, os.path.join(MODELS_DIR, 'encoder.pkl'))

    # Step 4a: Yield Prediction
    yield_model, yield_scaler, yield_features, yield_results = run_yield_prediction(df, feature_cols)
    
    # Step 4b: Crop Recommendation
    crop_model, crop_scaler, crop_features, crop_results = run_crop_recommendation(df, feature_cols)
    # Step 5: Ablation Study
    ablation_results = run_ablation_study(df)

    # Save scaler.pkl (combine scalers and features for easy loading)
    joblib.dump({
        'yield_scaler': yield_scaler, 'yield_features': yield_features,
        'crop_scaler': crop_scaler, 'crop_features': crop_features
    }, os.path.join(MODELS_DIR, 'scaler.pkl'))

    # Final Summary
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE — SUMMARY")
    print("=" * 70)
    print(f"\n  Yield Prediction Best R²:   {max(r['R2'] for r in yield_results.values()):.4f}")
    print(f"  Crop Recommendation Best F1: {max(r.get('F1', 0) for r in crop_results.values()):.4f}")
    print(f"\n  Saved models to: {MODELS_DIR}")
    print(f"  Saved results to: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
