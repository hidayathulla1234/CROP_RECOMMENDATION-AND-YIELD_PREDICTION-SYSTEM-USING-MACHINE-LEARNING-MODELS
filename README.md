# 🌾 Explainable Ensemble Learning Framework for Crop Recommendation, Yield Forecasting, and Profitability Assessment Using Multi-Source Agricultural Data

<p align="center">
  <img src="assets/system_overview.png" width="100%">
</p>

## 📖 Overview

This project presents an intelligent agricultural decision-support system that combines **Crop Recommendation**, **Yield Prediction**, and **Profitability Estimation** using Ensemble Machine Learning techniques. The framework utilizes multi-source agricultural data including soil characteristics, climate conditions, irrigation information, and historical yield records to assist farmers in making data-driven decisions.

---

## 🎯 Key Features

- 🌱 Crop Recommendation System
- 📈 Yield Prediction Model
- 💰 Profitability Assessment
- 🤖 Ensemble Learning Framework
- 🔍 Explainable AI using SHAP
- 📊 Cross-Validation and Ablation Study
- 🌐 FastAPI-based Web Application

---

## 🏗️ System Architecture

The proposed framework consists of:

1. **Input Layer** – Soil, climate, irrigation, and historical agricultural data.
2. **Data Preprocessing** – Missing value treatment, outlier handling, encoding, and scaling.
3. **Feature Engineering** – Creation of domain-specific agricultural features.
4. **Machine Learning Models**
   - Crop Recommendation (Classification)
   - Yield Prediction (Regression)
   - Profitability Estimation
5. **Decision Support Output** – Recommended crops, predicted yield, and estimated profit.
6. **Web Interface** – Interactive dashboard powered by FastAPI.

---

## 🤖 Machine Learning Models

### Classification Models
- Random Forest
- Extra Trees
- XGBoost
- LightGBM
- CatBoost
- Voting Ensemble
- Stacking Ensemble

### Regression Models
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor
- Neural Network (PyTorch)

---

## 📊 Dataset

The dataset contains agricultural observations collected from multiple sources across Odisha, India.

| Attribute | Value |
|------------|---------|
| Records | 1623 |
| Districts | 30 |
| Years | 2015–2025 |
| Features | Soil, Climate, Irrigation, Yield |
| Targets | Crop Type, Yield |

Each record represents a **District–Crop–Season–Year** agricultural observation.

---

## 🛠️ Technology Stack

- Python
- Scikit-Learn
- XGBoost
- LightGBM
- CatBoost
- PyTorch
- Optuna
- SHAP
- FastAPI
- React.js

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/agri-smart-framework.git
cd agri-smart-framework
pip install -r requirements.txt
```

### Run Backend

```bash
uvicorn app:app --reload
```

### Run Frontend

```bash
npm install
npm run dev
```

---

## 📈 Outputs

The system provides:

- Recommended Crop
- Alternative Crop Suggestions
- Predicted Yield
- Estimated Profit
- SHAP-Based Explanations
- Agricultural Insights

---
