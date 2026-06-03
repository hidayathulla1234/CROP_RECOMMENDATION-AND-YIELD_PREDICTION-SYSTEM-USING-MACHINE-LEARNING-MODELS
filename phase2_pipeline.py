"""

============================================================================
PHASE 2 — ULTRA MODEL IMPROVEMENT
Advanced Feature Engineering + Models + Ensemble + Validation
============================================================================

"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import json, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split, cross_val_score,
    StratifiedKFold, KFold, learning_curve)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report)
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    GradientBoostingRegressor, StackingRegressor, StackingClassifier,
    VotingRegressor, VotingClassifier)
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif, RFE
from sklearn.linear_model import Ridge, LogisticRegression

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
import shap
import torch, torch.nn as nn

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(os.path.dirname(BASE_DIR), 'smart_farming_final_dataset.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR= os.path.join(BASE_DIR, 'results')
TEST_DIR   = os.path.join(RESULTS_DIR, 'testing')
for d in [MODELS_DIR, RESULTS_DIR, TEST_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================================
# PHASE 1 — ADVANCED FEATURE ENGINEERING
# ============================================================================
def load_and_preprocess():
    print("=" * 70)
    print("  PHASE 2 — ULTRA MODEL IMPROVEMENT")
    print("=" * 70)
    print("\n[Phase 1] Advanced Feature Engineering")

    df = pd.read_csv(DATA_PATH)

    # Handle missing
    for c in df.select_dtypes(include=np.number).columns:
        df[c].fillna(df[c].median(), inplace=True)
    for c in df.select_dtypes(include='object').columns:
        df[c].fillna(df[c].mode()[0], inplace=True)

    # --- ORIGINAL FEATURES ---
    df['Nutrient_Ratio']       = (df['Nitrogen'] + df['Phosphorus'] + df['Potassium']) / 3
    df['Climate_Index']        = (df['Rainfall'] + df['Temperature'] + df['Humidity']) / 3
    df['Soil_Health']          = (df['Soil_pH'] + df['Soil_Fertility_Index']) / 2
    df['Water_Index']          = (df['Water_Stress_Index'] + df['Irrigation_Factor']) / 2
    df['Yield_Trend']          = df['Previous_Yield'] + df['Yield_Change']
    df['Fertilizer_Efficiency']= df['Yield'] / df['Fertilizer_kg_per_ha'].replace(0, np.nan)
    df['Fertilizer_Efficiency'].fillna(0, inplace=True)

    # --- PHASE 2 NEW FEATURES ---
    df['NPK_Ratio']                  = df['Nitrogen'] / (df['Phosphorus'] + df['Potassium'] + 1)
    df['Rainfall_Temperature_Index'] = df['Rainfall'] * df['Temperature']
    df['Soil_Moisture_Index']        = df['Humidity'] * df['Rainfall']
    df['Fertilizer_Interaction']     = df['Fertilizer_kg_per_ha'] * df['Soil_Fertility_Index']
    df['Climate_Risk_Index']         = df['Weather_Score'] * df['Rainfall_Score']
    df['Water_Availability_Index']   = df['Irrigation_Factor'] * df['Rainfall_Factor']

    # --- ENCODINGS ---
    ordinal_maps = {
        'Rainfall_Distribution': {'Low': 0, 'Moderate': 1, 'Good': 2, 'Heavy': 3},
        'Water_Stress_Level':    {'Low Stress': 0, 'Medium Stress': 1, 'High Stress': 2},
    }
    for col, mapping in ordinal_maps.items():
        df[col + '_encoded'] = df[col].map(mapping).fillna(0).astype(int)

    label_encoders = {}
    for c in ['District', 'State', 'Soil_Type', 'Irrigation', 'Season',
              'Crop_Category', 'Weather_Forecast']:
        le = LabelEncoder()
        df[c + '_encoded'] = le.fit_transform(df[c].astype(str))
        label_encoders[c] = le

    # --- INTERACTION FEATURES (need encodings first) ---
    df['Crop_Season_Interaction'] = df['Crop_Category_encoded'] * df['Season_encoded']
    # District-Season yield average (target encoding style)
    df['District_Season_Yield_Avg'] = df.groupby(
        ['District_encoded', 'Season_encoded'])['Yield'].transform('mean')

    # Target encoding for Crops
    le_crops = LabelEncoder()
    df['Crops_encoded'] = le_crops.fit_transform(df['Crops'])
    label_encoders['Crops'] = le_crops

    new_feats = ['NPK_Ratio', 'Rainfall_Temperature_Index', 'Soil_Moisture_Index',
                 'Fertilizer_Interaction', 'Climate_Risk_Index',
                 'Water_Availability_Index', 'Crop_Season_Interaction',
                 'District_Season_Yield_Avg']
    print(f"  Created {len(new_feats)} new Phase-2 features: {new_feats}")

    # --- OUTLIER REMOVAL ---
    initial = len(df)
    for c in ['Yield', 'Rainfall', 'Temperature', 'Fertilizer_kg_per_ha',
              'Nitrogen', 'Phosphorus', 'Potassium', 'Humidity', 'Area']:
        if c in df.columns and df[c].dtype in [np.float64, np.int64]:
            Q1, Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[c] >= Q1 - 1.5 * IQR) & (df[c] <= Q3 + 1.5 * IQR)]
    df = df.reset_index(drop=True)
    # Re-encode Crops after outlier removal
    le_crops2 = LabelEncoder()
    df['Crops_encoded'] = le_crops2.fit_transform(df['Crops'])
    label_encoders['Crops'] = le_crops2
    print(f"  Outlier removal: {initial} → {len(df)} rows")

    joblib.dump(label_encoders, os.path.join(MODELS_DIR, 'encoder.pkl'))
    print(f"  Saved encoder.pkl")
    return df, label_encoders


def get_feature_cols(df):
    exclude = {'Yield', 'Production', 'Crops', 'Crops_encoded',
               'Fertilizer_Efficiency', 'Yield_Efficiency',
               'District', 'State', 'Soil_Type', 'Irrigation', 'Season',
               'Crop_Category', 'Weather_Forecast', 'Rainfall_Distribution',
               'Water_Stress_Level'}
    return [c for c in df.columns if c not in exclude
            and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]


def select_features(X, y, task='regression', n=22):
    if task == 'regression':
        mi = mutual_info_regression(X, y, random_state=42)
    else:
        mi = mutual_info_classif(X, y, random_state=42)
    mi_scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)

    # Correlation drop
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = [c for c in upper.columns if any(upper[c] > 0.95)]

    top_mi = mi_scores[~mi_scores.index.isin(high_corr)].head(n).index.tolist()
    print(f"  Selected {len(top_mi)} features ({task}). Top 5: {top_mi[:5]}")
    return top_mi

# ============================================================================
# PYTORCH NEURAL NETWORK
# ============================================================================
class MLP(nn.Module):
    def __init__(self, inp, out, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h, h//2), nn.BatchNorm1d(h//2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h//2, h//4), nn.BatchNorm1d(h//4), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h//4, out)
        )
    def forward(self, x): return self.net(x)


def train_mlp(X_tr, y_tr, X_te, y_te, task='regression', epochs=200):
    Xtr = torch.FloatTensor(X_tr.values if hasattr(X_tr,'values') else X_tr)
    Xte = torch.FloatTensor(X_te.values if hasattr(X_te,'values') else X_te)
    if task == 'regression':
        Ytr = torch.FloatTensor(y_tr.values if hasattr(y_tr,'values') else y_tr).unsqueeze(1)
        odim, criterion = 1, nn.MSELoss()
    else:
        n_cls = int(y_tr.max()) + 1
        Ytr = torch.LongTensor(y_tr.values if hasattr(y_tr,'values') else y_tr)
        odim, criterion = n_cls, nn.CrossEntropyLoss()

    model = MLP(Xtr.shape[1], odim)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_loss, patience, no_improve = 1e9, 20, 0
    best_state = None

    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        loss = criterion(model(Xtr), Ytr)
        loss.backward(); opt.step(); sched.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(Xte)
        preds = out.squeeze().numpy() if task == 'regression' else out.argmax(1).numpy()
    return preds, model

# ============================================================================
# PHASE 2 — ADVANCED MODEL TRAINING
# ============================================================================
def optuna_tune(X_tr, y_tr, model_type='xgb', task='regression', n_trials=40):
    def objective(trial):
        if model_type == 'xgb':
            p = {'n_estimators': trial.suggest_int('n', 200, 800),
                 'max_depth': trial.suggest_int('d', 3, 10),
                 'learning_rate': trial.suggest_float('lr', 0.005, 0.3, log=True),
                 'subsample': trial.suggest_float('ss', 0.5, 1.0),
                 'colsample_bytree': trial.suggest_float('cs', 0.5, 1.0),
                 'reg_alpha': trial.suggest_float('ra', 1e-8, 5.0, log=True),
                 'reg_lambda': trial.suggest_float('rl', 1e-8, 5.0, log=True),
                 'min_child_weight': trial.suggest_int('mcw', 1, 10),
                 'random_state': 42, 'verbosity': 0}
            mdl = (xgb.XGBRegressor if task=='regression' else xgb.XGBClassifier)(**p, eval_metric='mlogloss' if task!='regression' else None)
        elif model_type == 'lgbm':
            p = {'n_estimators': trial.suggest_int('n', 200, 800),
                 'max_depth': trial.suggest_int('d', 3, 12),
                 'learning_rate': trial.suggest_float('lr', 0.005, 0.3, log=True),
                 'num_leaves': trial.suggest_int('nl', 15, 200),
                 'subsample': trial.suggest_float('ss', 0.5, 1.0),
                 'colsample_bytree': trial.suggest_float('cs', 0.5, 1.0),
                 'reg_alpha': trial.suggest_float('ra', 1e-8, 5.0, log=True),
                 'reg_lambda': trial.suggest_float('rl', 1e-8, 5.0, log=True),
                 'random_state': 42, 'verbose': -1}
            mdl = (lgb.LGBMRegressor if task=='regression' else lgb.LGBMClassifier)(**p)
        else:  # catboost
            p = {'iterations': trial.suggest_int('it', 200, 800),
                 'depth': trial.suggest_int('d', 3, 10),
                 'learning_rate': trial.suggest_float('lr', 0.005, 0.3, log=True),
                 'l2_leaf_reg': trial.suggest_float('l2', 1e-8, 10.0, log=True),
                 'bagging_temperature': trial.suggest_float('bt', 0, 1),
                 'random_seed': 42, 'verbose': 0}
            mdl = (cb.CatBoostRegressor if task=='regression' else cb.CatBoostClassifier)(**p)

        scoring = 'r2' if task=='regression' else 'f1_weighted'
        cv = KFold(5, shuffle=True, random_state=42) if task=='regression' \
             else StratifiedKFold(5, shuffle=True, random_state=42)
        return cross_val_score(mdl, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1).mean()

    study = optuna.create_study(direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    print(f"    Optuna {model_type.upper()} best={study.best_value:.4f} ({n_trials} trials)")
    return study.best_params, study.best_value


def build_all_regression_models(X_tr, y_tr, X_te, y_te, X_all, y_all):
    print("\n[Phase 2] Advanced Regression Model Training")
    results, trained = {}, {}

    def eval_reg(name, mdl, preds=None):
        if preds is None:
            mdl.fit(X_tr, y_tr)
            preds = mdl.predict(X_te)
        r2   = r2_score(y_te, preds)
        rmse = np.sqrt(mean_squared_error(y_te, preds))
        mae  = mean_absolute_error(y_te, preds)
        cv   = cross_val_score(mdl, X_all, y_all, cv=KFold(5, shuffle=True, random_state=42),
                               scoring='r2', n_jobs=1).mean() if hasattr(mdl,'fit') else r2
        results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'CV_R2': cv}
        print(f"  {name:28s} | R²={r2:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f} | CV_R²={cv:.4f}")
        return mdl

    # Base models
    eval_reg('ExtraTrees',   ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=1))
    eval_reg('GradientBoosting', __import__('sklearn.ensemble', fromlist=['GradientBoostingRegressor']).GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.08, random_state=42))

    # Neural Net
    nn_preds, nn_model = train_mlp(X_tr, y_tr, X_te, y_te, 'regression')
    r2 = r2_score(y_te, nn_preds)
    results['NeuralNet_v2'] = {'R2': r2, 'RMSE': np.sqrt(mean_squared_error(y_te, nn_preds)),
                               'MAE': mean_absolute_error(y_te, nn_preds), 'CV_R2': r2}
    print(f"  {'NeuralNet_v2':28s} | R²={r2:.4f}")

    # Optuna tuned
    print("  --- Optuna Tuning (40+40+30 trials) ---")
    for mtype, n_t in [('xgb',40), ('lgbm',40), ('catboost',30)]:
        params, score = optuna_tune(X_tr, y_tr, mtype, 'regression', n_t)
        if mtype == 'xgb':
            m = xgb.XGBRegressor(**{k.replace('n','n_estimators').replace('d','max_depth')
                .replace('lr','learning_rate').replace('ss','subsample')
                .replace('cs','colsample_bytree').replace('ra','reg_alpha')
                .replace('rl','reg_lambda').replace('mcw','min_child_weight'): v
                for k,v in params.items()}, random_state=42, verbosity=0)
            name = 'XGBoost_Optuna'
        elif mtype == 'lgbm':
            m = lgb.LGBMRegressor(**{k.replace('n','n_estimators').replace('d','max_depth')
                .replace('lr','learning_rate').replace('nl','num_leaves')
                .replace('ss','subsample').replace('cs','colsample_bytree')
                .replace('ra','reg_alpha').replace('rl','reg_lambda'): v
                for k,v in params.items()}, random_state=42, verbose=-1)
            name = 'LightGBM_Optuna'
        else:
            m = cb.CatBoostRegressor(**{k.replace('it','iterations').replace('d','depth')
                .replace('lr','learning_rate').replace('l2','l2_leaf_reg')
                .replace('bt','bagging_temperature'): v
                for k,v in params.items()}, random_seed=42, verbose=0)
            name = 'CatBoost_Optuna'
        trained[name] = eval_reg(name, m)

    # Phase 3 — Ensembles
    print("  --- Ultra Ensembles ---")
    xm = trained.get('XGBoost_Optuna',  xgb.XGBRegressor(n_estimators=400, random_state=42, verbosity=0))
    lm = trained.get('LightGBM_Optuna', lgb.LGBMRegressor(n_estimators=400, random_state=42, verbose=-1))
    cm = trained.get('CatBoost_Optuna', cb.CatBoostRegressor(iterations=400, random_seed=42, verbose=0))

    # Weighted Voting Ensemble
    voting = VotingRegressor([('xgb', xm), ('lgbm', lm), ('cb', cm)],
                              weights=[0.35, 0.30, 0.35])
    voting.fit(X_tr, y_tr)
    eval_reg('WeightedVoting', voting)
    trained['WeightedVoting'] = voting

    # Stacking Ensemble
    stacking = StackingRegressor(
        estimators=[('xgb', xgb.XGBRegressor(n_estimators=300, random_state=42, verbosity=0)),
                    ('lgbm', lgb.LGBMRegressor(n_estimators=300, random_state=42, verbose=-1)),
                    ('cb',   cb.CatBoostRegressor(iterations=300, random_seed=42, verbose=0)),
                    ('et',   ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=1))],
        final_estimator=Ridge(alpha=0.5), cv=5)
    stacking.fit(X_tr, y_tr)
    eval_reg('StackingEnsemble', stacking)
    trained['StackingEnsemble'] = stacking

    # Blended model
    p_xgb  = xm.predict(X_te)
    p_lgbm = lm.predict(X_te)
    p_cb   = cm.predict(X_te)
    blend  = 0.35*p_xgb + 0.30*p_lgbm + 0.35*p_cb
    r2b = r2_score(y_te, blend)
    results['BlendedModel'] = {'R2': r2b, 'RMSE': np.sqrt(mean_squared_error(y_te, blend)),
                               'MAE': mean_absolute_error(y_te, blend), 'CV_R2': r2b}
    print(f"  {'BlendedModel':28s} | R²={r2b:.4f}")

    best = max(results, key=lambda k: results[k]['R2'])
    print(f"\n  *** BEST YIELD MODEL: {best} (R²={results[best]['R2']:.4f}) ***")

    best_model = trained.get(best, stacking)
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'yield_model.pkl'))
    print(f"  Saved yield_model.pkl")
    return best_model, results, trained


def build_all_classification_models(X_tr, y_tr, X_te, y_te, X_all, y_all, le_crops):
    print("\n[Phase 2] Advanced Classification Model Training")
    results, trained = {}, {}

    def eval_clf(name, mdl, preds=None):
        if preds is None:
            mdl.fit(X_tr, y_tr)
            preds = mdl.predict(X_te)
        acc  = accuracy_score(y_te, preds)
        prec = precision_score(y_te, preds, average='weighted', zero_division=0)
        rec  = recall_score(y_te, preds, average='weighted', zero_division=0)
        f1   = f1_score(y_te, preds, average='weighted', zero_division=0)
        cv   = cross_val_score(mdl, X_all, y_all,
                cv=StratifiedKFold(5, shuffle=True, random_state=42),
                scoring='f1_weighted', n_jobs=1).mean() if hasattr(mdl,'fit') else f1
        results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'CV_F1': cv}
        print(f"  {name:28s} | Acc={acc:.4f} | F1={f1:.4f} | CV_F1={cv:.4f}")
        return mdl

    # Base
    eval_clf('ExtraTrees', ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=1))

    # Neural Net
    nn_preds, nn_model = train_mlp(X_tr, y_tr, X_te, y_te, 'classification')
    acc = accuracy_score(y_te, nn_preds)
    f1  = f1_score(y_te, nn_preds, average='weighted', zero_division=0)
    results['NeuralNet_v2'] = {'Accuracy': acc, 'F1': f1, 'CV_F1': f1}
    print(f"  {'NeuralNet_v2':28s} | Acc={acc:.4f} | F1={f1:.4f}")

    # Optuna
    print("  --- Optuna Tuning (40+40+30 trials) ---")
    for mtype, n_t in [('xgb',40), ('lgbm',40), ('catboost',30)]:
        params, score = optuna_tune(X_tr, y_tr, mtype, 'classification', n_t)
        if mtype == 'xgb':
            m = xgb.XGBClassifier(**{k.replace('n','n_estimators').replace('d','max_depth')
                .replace('lr','learning_rate').replace('ss','subsample')
                .replace('cs','colsample_bytree').replace('ra','reg_alpha')
                .replace('rl','reg_lambda').replace('mcw','min_child_weight'): v
                for k,v in params.items()}, random_state=42, verbosity=0, eval_metric='mlogloss')
            name = 'XGBoost_Optuna'
        elif mtype == 'lgbm':
            m = lgb.LGBMClassifier(**{k.replace('n','n_estimators').replace('d','max_depth')
                .replace('lr','learning_rate').replace('nl','num_leaves')
                .replace('ss','subsample').replace('cs','colsample_bytree')
                .replace('ra','reg_alpha').replace('rl','reg_lambda'): v
                for k,v in params.items()}, random_state=42, verbose=-1)
            name = 'LightGBM_Optuna'
        else:
            m = cb.CatBoostClassifier(**{k.replace('it','iterations').replace('d','depth')
                .replace('lr','learning_rate').replace('l2','l2_leaf_reg')
                .replace('bt','bagging_temperature'): v
                for k,v in params.items()}, random_seed=42, verbose=0)
            name = 'CatBoost_Optuna'
        trained[name] = eval_clf(name, m)

    # Ensembles
    print("  --- Ultra Ensembles ---")
    xm = trained.get('XGBoost_Optuna',  xgb.XGBClassifier(n_estimators=400, random_state=42, verbosity=0, eval_metric='mlogloss'))
    lm = trained.get('LightGBM_Optuna', lgb.LGBMClassifier(n_estimators=400, random_state=42, verbose=-1))
    cm = trained.get('CatBoost_Optuna', cb.CatBoostClassifier(iterations=400, random_seed=42, verbose=0))

    # Weighted Voting
    voting = VotingClassifier([('xgb', xm), ('lgbm', lm), ('cb', cm)],
                               voting='soft', weights=[0.35, 0.30, 0.35])
    voting.fit(X_tr, y_tr)
    eval_clf('WeightedVoting', voting)
    trained['WeightedVoting'] = voting

    # Stacking
    stacking = StackingClassifier(
        estimators=[('xgb',  xgb.XGBClassifier(n_estimators=300, random_state=42, verbosity=0, eval_metric='mlogloss')),
                    ('lgbm', lgb.LGBMClassifier(n_estimators=300, random_state=42, verbose=-1)),
                    ('cb',   cb.CatBoostClassifier(iterations=300, random_seed=42, verbose=0)),
                    ('et',   ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=1))],
        final_estimator=LogisticRegression(max_iter=2000, C=1.0, random_state=42), cv=5)
    stacking.fit(X_tr, y_tr)
    eval_clf('StackingEnsemble', stacking)
    trained['StackingEnsemble'] = stacking

    best = max(results, key=lambda k: results[k].get('F1', 0))
    print(f"\n  *** BEST CROP MODEL: {best} (F1={results[best]['F1']:.4f}) ***")

    best_model = trained.get(best, stacking)

    # Confusion matrix
    preds_final = best_model.predict(X_te)
    cm_matrix   = confusion_matrix(y_te, preds_final)
    class_names = le_crops.classes_ if hasattr(le_crops, 'classes_') else None
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title('Phase 2 — Confusion Matrix (Crop Recommendation)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_phase2.png'), dpi=150)
    plt.close()

    print("\n  Classification Report:")
    print(classification_report(y_te, preds_final,
          target_names=class_names, zero_division=0))

    joblib.dump(best_model, os.path.join(MODELS_DIR, 'crop_model.pkl'))
    print(f"  Saved crop_model.pkl")
    return best_model, results, trained


# ============================================================================
# PHASE 7 — MODEL TESTING & VALIDATION
# ============================================================================
def learning_curve_analysis(model, X, y, task='regression'):
    print(f"\n[Phase 7] Learning Curve Analysis ({task})")
    scoring = 'r2' if task == 'regression' else 'f1_weighted'
    cv      = KFold(5, shuffle=True, random_state=42) if task == 'regression' \
              else StratifiedKFold(5, shuffle=True, random_state=42)
    sizes   = np.linspace(0.1, 1.0, 8)

    try:
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, scoring=scoring, train_sizes=sizes, n_jobs=1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_sizes, train_scores.mean(axis=1), 'b-o', label='Train Score')
        ax.plot(train_sizes, val_scores.mean(axis=1),   'r-o', label='Validation Score')
        ax.fill_between(train_sizes,
            train_scores.mean(1)-train_scores.std(1),
            train_scores.mean(1)+train_scores.std(1), alpha=0.1, color='b')
        ax.fill_between(train_sizes,
            val_scores.mean(1)-val_scores.std(1),
            val_scores.mean(1)+val_scores.std(1), alpha=0.1, color='r')
        ax.set_xlabel('Training Examples')
        ax.set_ylabel(scoring.upper())
        ax.set_title(f'Learning Curve — {task.title()}')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(TEST_DIR, f'learning_curve_{task}.png'), dpi=150)
        plt.close()

        gap = (train_scores.mean(axis=1)[-1] - val_scores.mean(axis=1)[-1])
        overfit = "⚠️  OVERFIT" if gap > 0.05 else "✅ Good generalisation"
        print(f"  Train/Val gap = {gap:.4f}  →  {overfit}")
        return {
            'train_score_final': float(train_scores.mean(axis=1)[-1]),
            'val_score_final':   float(val_scores.mean(axis=1)[-1]),
            'gap':               float(gap),
            'overfit_detected':  gap > 0.05
        }
    except Exception as e:
        print(f"  Learning curve error: {e}")
        return {}


def bias_variance_analysis(model, X_tr, y_tr, X_te, y_te, task='regression'):
    print(f"\n[Phase 7] Bias-Variance Analysis ({task})")
    scoring = 'r2' if task == 'regression' else 'f1_weighted'
    cv = KFold(5, shuffle=True, random_state=42) if task == 'regression' \
         else StratifiedKFold(5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1)

    if task == 'regression':
        tr_score = r2_score(y_tr, model.predict(X_tr))
        te_score = r2_score(y_te, model.predict(X_te))
    else:
        tr_score = f1_score(y_tr, model.predict(X_tr), average='weighted', zero_division=0)
        te_score = f1_score(y_te, model.predict(X_te), average='weighted', zero_division=0)

    bias    = 1 - te_score
    variance= abs(tr_score - te_score)
    print(f"  Train {scoring}:  {tr_score:.4f}")
    print(f"  Test  {scoring}:  {te_score:.4f}")
    print(f"  CV    {scoring}:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Bias:            {bias:.4f}")
    print(f"  Variance (gap):  {variance:.4f}")
    print(f"  Status: {'⚠️ High Variance' if variance>0.05 else '✅ Balanced'}")

    return {'train': tr_score, 'test': te_score, 'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()), 'bias': bias, 'variance': variance}


def shap_analysis(model, X, feature_names, task='regression'):
    try:
        explainer    = shap.TreeExplainer(model)
        shap_values  = explainer.shap_values(X[:min(300,len(X))])
        plt.figure(figsize=(12, 8))
        sv = shap_values[0] if isinstance(shap_values, list) else shap_values
        shap.summary_plot(sv, X[:min(300,len(X))],
                          feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(TEST_DIR, f'shap_phase2_{task}.png'), dpi=150, bbox_inches='tight')
        plt.close('all')
        print(f"  SHAP plot saved → results/testing/shap_phase2_{task}.png")
    except Exception as e:
        print(f"  SHAP: {e}")


# ============================================================================
# SAVE SCALERS AS scaler.pkl (unified)
# ============================================================================
def save_production_models(yield_scaler, yield_feats, crop_scaler, crop_feats):
    joblib.dump({'yield': yield_scaler, 'crop': crop_scaler},
                os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump({'yield': yield_feats,  'crop': crop_feats},
                os.path.join(MODELS_DIR, 'feature_lists.pkl'))
    print(f"  Saved scaler.pkl, feature_lists.pkl")


# ============================================================================
# MAIN
# ============================================================================

def main():
    df, label_encoders = load_and_preprocess()
    feat_cols = get_feature_cols(df)

    # --- YIELD PREDICTION ---
    Xy = df[feat_cols].copy()
    yy = df['Yield'].copy()
    sel_y = select_features(Xy, yy, 'regression', n=22)
    Xy = Xy[sel_y]
    scaler_y = StandardScaler()
    Xy_s = pd.DataFrame(scaler_y.fit_transform(Xy), columns=Xy.columns)
    Xtr_y, Xte_y, ytr_y, yte_y = train_test_split(Xy_s, yy, test_size=0.2, random_state=42)

    best_yield, yield_results, yield_trained = build_all_regression_models(
        Xtr_y, ytr_y, Xte_y, yte_y, Xy_s, yy)

    # --- CROP RECOMMENDATION ---
    Xc = df[feat_cols].copy()
    yc = df['Crops_encoded'].copy()
    sel_c = select_features(Xc, yc, 'classification', n=22)
    Xc = Xc[sel_c]
    scaler_c = StandardScaler()
    Xc_s = pd.DataFrame(scaler_c.fit_transform(Xc), columns=Xc.columns)
    Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(
        Xc_s, yc, test_size=0.2, random_state=42, stratify=yc)

    best_crop, crop_results, crop_trained = build_all_classification_models(
        Xtr_c, ytr_c, Xte_c, yte_c, Xc_s, yc, label_encoders['Crops'])

    # --- PHASE 7: VALIDATION ---
    print("\n[Phase 7] Model Testing & Validation")
    lc_y = learning_curve_analysis(best_yield, Xy_s, yy, 'regression')
    bv_y = bias_variance_analysis(best_yield, Xtr_y, ytr_y, Xte_y, yte_y, 'regression')
    shap_analysis(best_yield if hasattr(best_yield,'feature_importances_') else
                  yield_trained.get('CatBoost_Optuna', yield_trained.get('XGBoost_Optuna')),
                  Xte_y, sel_y, 'regression')

    lc_c = learning_curve_analysis(best_crop, Xc_s, yc, 'classification')
    bv_c = bias_variance_analysis(best_crop, Xtr_c, ytr_c, Xte_c, yte_c, 'classification')
    shap_analysis(crop_trained.get('CatBoost_Optuna', crop_trained.get('XGBoost_Optuna')),
                  Xte_c, sel_c, 'classification')

    # --- PHASE 8: SAVE PRODUCTION MODELS ---
    print("\n[Phase 8] Saving Production Models")
    save_production_models(scaler_y, sel_y, scaler_c, sel_c)

    # --- SAVE METRICS ---
    all_metrics = {
        'yield_models':   {k: {kk: round(vv,6) for kk,vv in v.items()} for k,v in yield_results.items()},
        'crop_models':    {k: {kk: round(vv,6) for kk,vv in v.items()} for k,v in crop_results.items()},
        'yield_learning': lc_y, 'yield_bv': bv_y,
        'crop_learning':  lc_c, 'crop_bv':  bv_c,
    }
    with open(os.path.join(TEST_DIR, 'phase2_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # --- FINAL SUMMARY ---
    best_r2 = max(r['R2'] for r in yield_results.values())
    best_f1 = max(r.get('F1',0) for r in crop_results.values())
    print("\n" + "=" * 70)
    print("  PHASE 2 COMPLETE — FINAL RESULTS")
    print("=" * 70)
    print(f"  Yield Prediction  Best R²  : {best_r2:.4f}")
    print(f"  Crop Recommender  Best F1  : {best_f1:.4f}")
    print(f"  Models saved to  : {MODELS_DIR}")
    print(f"  Results saved to : {TEST_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
