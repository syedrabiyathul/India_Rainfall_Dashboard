"""
Machine Learning Engine for Rainfall Prediction and Model Evaluation.
Implements Feature Engineering, Random Forest Regressor & Classifier,
Gradient Boosting, and complete diagnostic evaluation metrics.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
FEATURE_COLS = ['Jan-Feb', 'Mar-May', 'LAG_1', 'LAG_2', 'ROLL_MEAN_3', 'ROLL_MEAN_5', 'NORMAL_ANNUAL']
CLASSES = ['Deficient', 'Normal', 'Excess']

def engineer_features(df_sub: pd.DataFrame) -> pd.DataFrame:
    """Creates lagged and rolling features for time-series rainfall modeling."""
    df = df_sub.sort_values(by=['SUBDIVISION', 'YEAR']).copy()
    
    # Lagged features per subdivision
    df['LAG_1'] = df.groupby('SUBDIVISION')['ANNUAL'].shift(1)
    df['LAG_2'] = df.groupby('SUBDIVISION')['ANNUAL'].shift(2)
    
    # Rolling averages over previous years
    df['ROLL_MEAN_3'] = df.groupby('SUBDIVISION')['ANNUAL'].shift(1).rolling(3).mean()
    df['ROLL_MEAN_5'] = df.groupby('SUBDIVISION')['ANNUAL'].shift(1).rolling(5).mean()
    
    # Impute initial NaNs using subdivision climatological normal
    df['LAG_1'] = df['LAG_1'].fillna(df['NORMAL_ANNUAL'])
    df['LAG_2'] = df['LAG_2'].fillna(df['NORMAL_ANNUAL'])
    df['ROLL_MEAN_3'] = df['ROLL_MEAN_3'].fillna(df['NORMAL_ANNUAL'])
    df['ROLL_MEAN_5'] = df['ROLL_MEAN_5'].fillna(df['NORMAL_ANNUAL'])
    
    return df

def train_models(df_sub: pd.DataFrame):
    """
    Trains Regression and Classification models on historical rainfall data.
    Saves models and metrics to the models/ directory.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = engineer_features(df_sub)
    
    X = df[FEATURE_COLS]
    y_reg = df['ANNUAL']
    y_clf = df['CATEGORY_3CLASS']
    
    # 1. Train Regression Model (Random Forest Regressor)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    reg_model = RandomForestRegressor(
        n_estimators=150, max_depth=12, min_samples_split=4, random_state=42, n_jobs=-1
    )
    reg_model.fit(Xr_train, yr_train)
    
    yr_pred = reg_model.predict(Xr_test)
    r2 = r2_score(yr_test, yr_pred)
    rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
    mae = mean_absolute_error(yr_test, yr_pred)
    
    # Regression Feature Importance
    reg_feat_imp = dict(zip(FEATURE_COLS, [round(float(v), 4) for v in reg_model.feature_importances_]))
    
    # 2. Train Classification Model (Random Forest with class weights)
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    clf_model = RandomForestClassifier(
        n_estimators=150, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1
    )
    clf_model.fit(Xc_train, yc_train)
    
    yc_pred = clf_model.predict(Xc_test)
    
    # Also train comparison model: Gradient Boosting Classifier
    gb_model = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, random_state=42
    )
    gb_model.fit(Xc_train, yc_train)
    gb_pred = gb_model.predict(Xc_test)
    
    # Metrics for Primary Classifier (RF)
    acc = accuracy_score(yc_test, yc_pred)
    prec_macro = precision_score(yc_test, yc_pred, labels=CLASSES, average='macro', zero_division=0)
    rec_macro = recall_score(yc_test, yc_pred, labels=CLASSES, average='macro', zero_division=0)
    f1_macro = f1_score(yc_test, yc_pred, labels=CLASSES, average='macro', zero_division=0)
    
    prec_weighted = precision_score(yc_test, yc_pred, labels=CLASSES, average='weighted', zero_division=0)
    rec_weighted = recall_score(yc_test, yc_pred, labels=CLASSES, average='weighted', zero_division=0)
    f1_weighted = f1_score(yc_test, yc_pred, labels=CLASSES, average='weighted', zero_division=0)
    
    # Per-class metrics
    p_per_class = precision_score(yc_test, yc_pred, labels=CLASSES, average=None, zero_division=0)
    r_per_class = recall_score(yc_test, yc_pred, labels=CLASSES, average=None, zero_division=0)
    f_per_class = f1_score(yc_test, yc_pred, labels=CLASSES, average=None, zero_division=0)
    
    cm = confusion_matrix(yc_test, yc_pred, labels=CLASSES).tolist()
    clf_feat_imp = dict(zip(FEATURE_COLS, [round(float(v), 4) for v in clf_model.feature_importances_]))
    
    # Metrics for GB comparison
    gb_acc = accuracy_score(yc_test, gb_pred)
    gb_f1_macro = f1_score(yc_test, gb_pred, labels=CLASSES, average='macro', zero_division=0)
    gb_cm = confusion_matrix(yc_test, gb_pred, labels=CLASSES).tolist()
    
    metrics = {
        "regression": {
            "r2": round(r2, 4),
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "feature_importance": reg_feat_imp,
            "test_size": len(yr_test)
        },
        "classification": {
            "model_name": "Random Forest (Balanced)",
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "precision_weighted": round(prec_weighted, 4),
            "recall_weighted": round(rec_weighted, 4),
            "f1_weighted": round(f1_weighted, 4),
            "per_class": {
                c: {
                    "precision": round(float(p), 4),
                    "recall": round(float(r), 4),
                    "f1": round(float(f), 4)
                } for c, p, r, f in zip(CLASSES, p_per_class, r_per_class, f_per_class)
            },
            "confusion_matrix": cm,
            "classes": CLASSES,
            "feature_importance": clf_feat_imp,
            "comparison_model": {
                "name": "Gradient Boosting Classifier",
                "accuracy": round(gb_acc, 4),
                "f1_macro": round(gb_f1_macro, 4),
                "confusion_matrix": gb_cm
            }
        }
    }
    
    # Save artifacts
    joblib.dump(reg_model, os.path.join(MODEL_DIR, "rf_regressor.joblib"))
    joblib.dump(clf_model, os.path.join(MODEL_DIR, "rf_classifier.joblib"))
    joblib.dump(gb_model, os.path.join(MODEL_DIR, "gb_classifier.joblib"))
    joblib.dump(metrics, os.path.join(MODEL_DIR, "metrics.joblib"))
    
    return reg_model, clf_model, metrics

def load_or_train_models(df_sub: pd.DataFrame):
    """Loads existing models or triggers training if not found."""
    reg_path = os.path.join(MODEL_DIR, "rf_regressor.joblib")
    clf_path = os.path.join(MODEL_DIR, "rf_classifier.joblib")
    metrics_path = os.path.join(MODEL_DIR, "metrics.joblib")
    
    if os.path.exists(reg_path) and os.path.exists(clf_path) and os.path.exists(metrics_path):
        reg_model = joblib.load(reg_path)
        clf_model = joblib.load(clf_path)
        metrics = joblib.load(metrics_path)
        return reg_model, clf_model, metrics
    else:
        return train_models(df_sub)

def predict_scenario(
    reg_model, clf_model, normal_annual: float,
    jan_feb: float, mar_may: float, lag_1: float, lag_2: float,
    roll_3: float, roll_5: float
):
    """
    Executes dual-model inference for scenario simulation:
    - Continuous rainfall prediction (mm) via Regressor
    - Meteorological category classification via Classifier
    - Derived departure percentage & IMD classification
    """
    input_df = pd.DataFrame([{
        'Jan-Feb': jan_feb,
        'Mar-May': mar_may,
        'LAG_1': lag_1,
        'LAG_2': lag_2,
        'ROLL_MEAN_3': roll_3,
        'ROLL_MEAN_5': roll_5,
        'NORMAL_ANNUAL': normal_annual
    }])[FEATURE_COLS]
    
    # Regressor prediction
    pred_annual = float(reg_model.predict(input_df)[0])
    
    # Classification prediction & probabilities
    pred_category = str(clf_model.predict(input_df)[0])
    pred_probs = clf_model.predict_proba(input_df)[0]
    prob_dict = {cls: round(float(prob) * 100, 1) for cls, prob in zip(clf_model.classes_, pred_probs)}
    
    # Derived departure from normal
    dep_pct = ((pred_annual - normal_annual) / normal_annual) * 100
    
    # Model-derived IMD category from regression
    if dep_pct >= 60.0:
        imd_derived = "Large Excess"
    elif dep_pct >= 20.0:
        imd_derived = "Excess"
    elif dep_pct >= -19.0:
        imd_derived = "Normal"
    elif dep_pct >= -59.0:
        imd_derived = "Deficient"
    else:
        imd_derived = "Scanty"
        
    return {
        "predicted_annual_mm": round(pred_annual, 1),
        "normal_annual_mm": round(normal_annual, 1),
        "predicted_departure_pct": round(dep_pct, 1),
        "model_derived_category": imd_derived,
        "direct_classifier_category": pred_category,
        "class_probabilities": prob_dict,
        "confidence_level": max(prob_dict.values()),
        "is_model_derived": True
    }

if __name__ == "__main__":
    from data_loader import load_data
    df_s, _ = load_data()
    print("Training models...")
    reg, clf, m = train_models(df_s)
    print("Regression R2:", m['regression']['r2'])
    print("Classification Accuracy:", m['classification']['accuracy'])
    print("Classification F1 Macro:", m['classification']['f1_macro'])
