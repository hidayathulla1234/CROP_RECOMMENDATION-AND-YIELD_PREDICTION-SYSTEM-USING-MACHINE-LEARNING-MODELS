Ensemble Learning Framework for Crop Recommendation and Yield Forecasting Using Multi-Source Agricultural Data
Overview

This project presents an intelligent agricultural decision-support framework that combines crop recommendation, yield prediction, profitability assessment, and explainable artificial intelligence (XAI). The system utilizes ensemble machine learning techniques and multi-source agricultural datasets collected across districts and seasons to support data-driven farming decisions.

Features
Crop Recommendation using ensemble classification models
Crop Yield Prediction using ensemble regression models
Profitability Assessment for recommended crops
Advanced Feature Engineering
Hyperparameter Optimization using Optuna
Explainable AI using SHAP Analysis
Partial Dependence Plot (PDP) Visualization
Cross-Validation and Model Evaluation
Ablation Study for feature contribution analysis
Dataset

The dataset contains agricultural information collected from multiple districts between 2015 and 2025.

Dataset Characteristics
1623 observations
30 districts
Multiple crop categories
Climate variables
Soil parameters
Irrigation information
Fertilizer usage records
Historical yield information
Project Structure
project/
│
├── data/
│   └── smart_farming_final_dataset.csv
│
├── models/
│   ├── crop_model.pkl
│   ├── yield_model.pkl
│   ├── scaler.pkl
│   └── encoder.pkl
│
├── results/
│   ├── crop_metrics.json
│   ├── yield_metrics.json
│   ├── shap_summary_regression.png
│   ├── shap_summary_classification.png
│   ├── confusion_matrix.png
│   └── ablation_study.csv
│
├── src/
│   └── main.py
│
└── README.md
Machine Learning Models
Classification Models
Random Forest
XGBoost
LightGBM
CatBoost
Extra Trees
Voting Ensemble
Weighted Voting Ensemble
Stacking Ensemble
Neural Network
Regression Models
Random Forest Regressor
XGBoost Regressor
LightGBM Regressor
CatBoost Regressor
Gradient Boosting Regressor
Extra Trees Regressor
Voting Regressor
Weighted Voting Regressor
Stacking Regressor
Neural Network
Feature Engineering

The framework generates several agricultural indicators:

Nutrient Ratio
Climate Index
Soil Health Index
Water Index
NPK Ratio
Rainfall–Temperature Index
Soil Moisture Index
Fertilizer Interaction
Climate Risk Index
Water Availability Index
Evaluation Metrics
Crop Recommendation
Accuracy
Precision
Recall
F1-Score
Cross-Validation Accuracy
Yield Prediction
R² Score
RMSE
MAE
Cross-Validation R²
Explainability

The framework incorporates Explainable AI techniques:

SHAP Summary Plot
Feature Importance Analysis
Partial Dependence Plots (PDP)
Installation
git clone https://github.com/your-username/crop-recommendation-yield-prediction.git

cd crop-recommendation-yield-prediction

pip install -r requirements.txt
Run the Project
python main.py
Results

The proposed ensemble learning framework demonstrates strong performance in both crop recommendation and yield forecasting while maintaining interpretability through SHAP-based explanations.

Authors
Dr.Ishapathik Das
Shaik Hidayathulla
Sai Kalpana Pathlavath

License
This project is released for academic and research purposes.
