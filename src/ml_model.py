"""
Machine Learning Model module for India Rainfall Prediction.
Implements Random Forest Regressor, Classifier, Model Persistence to rainfall_model.pkl,
and real-time inference simulator.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from src.utils import MODELS_DIR, MODEL_PKL, FEATURE_COLS, CLASSES
from src.preprocessing import load_clean_data

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

def train_and_save_model(df_sub: pd.DataFrame = None):
    """
    Trains the primary Random Forest model and persists to models/rainfall_model.pkl.
    Also returns evaluation metrics dictionary.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    if df_sub is None:
        df_sub, _ = load_clean_data()
        
    df = engineer_features(df_sub)
    X = df[FEATURE_COLS]
    y_reg = df['ANNUAL']
    y_clf = df['CATEGORY_3CLASS']
    
    # 1. Regressor Split & Training
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    reg_model = RandomForestRegressor(
        n_estimators=150, max_depth=12, min_samples_split=4, random_state=42, n_jobs=-1
    )
    reg_model.fit(Xr_train, yr_train)
    y_reg_pred = reg_model.predict(Xr_test)
    
    r2 = r2_score(yr_test, y_reg_pred)
    rmse = np.sqrt(mean_squared_error(yr_test, y_reg_pred))
    mae = mean_absolute_error(yr_test, y_reg_pred)
    
    feat_imp = dict(zip(FEATURE_COLS, [round(float(v), 4) for v in reg_model.feature_importances_]))
    
    # 2. Classifier Split & Training
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    clf_model = RandomForestClassifier(
        n_estimators=150, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1
    )
    clf_model.fit(Xc_train, yc_train)
    y_clf_pred = clf_model.predict(Xc_test)
    
    acc = accuracy_score(yc_test, y_clf_pred)
    prec_macro = precision_score(yc_test, y_clf_pred, average='macro', zero_division=0)
    rec_macro = recall_score(yc_test, y_clf_pred, average='macro', zero_division=0)
    f1_macro = f1_score(yc_test, y_clf_pred, average='macro', zero_division=0)
    cm = confusion_matrix(yc_test, y_clf_pred, labels=CLASSES).tolist()
    
    report_dict = classification_report(yc_test, y_clf_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    
    metrics = {
        "regression": {
            "r2": round(float(r2), 4),
            "r2_score": round(float(r2), 4),
            "rmse": round(float(rmse), 2),
            "mae": round(float(mae), 2),
            "test_size": len(yr_test),
            "feature_importance": feat_imp
        },
        "classification": {
            "accuracy": round(float(acc), 4),
            "precision_macro": round(float(prec_macro), 4),
            "recall_macro": round(float(rec_macro), 4),
            "f1_macro": round(float(f1_macro), 4),
            "confusion_matrix": cm,
            "per_class": {
                c: {
                    "precision": round(report_dict[c]['precision'], 3),
                    "recall": round(report_dict[c]['recall'], 3),
                    "f1": round(report_dict[c]['f1-score'], 3)
                } for c in CLASSES if c in report_dict
            }
        }
    }
    metrics["regressor"] = metrics["regression"]
    metrics["classifier"] = metrics["classification"]
    
    # Save composite model payload
    model_payload = {
        "regressor": reg_model,
        "classifier": clf_model,
        "metrics": metrics,
        "feature_cols": FEATURE_COLS,
        "classes": CLASSES
    }
    
    joblib.dump(model_payload, MODEL_PKL)
    # Also save separate components for backward compatibility
    joblib.dump(reg_model, os.path.join(MODELS_DIR, "rf_regressor.joblib"))
    joblib.dump(clf_model, os.path.join(MODELS_DIR, "rf_classifier.joblib"))
    joblib.dump(metrics, os.path.join(MODELS_DIR, "metrics.joblib"))
    
    return model_payload

def load_model():
    """Loads the persisted rainfall model artifact."""
    if os.path.exists(MODEL_PKL):
        return joblib.load(MODEL_PKL)
    return train_and_save_model()

def predict_rainfall(
    subdivision: str,
    normal_annual: float,
    jan_feb: float,
    mar_may: float,
    lag_1: float,
    lag_2: float,
    roll_mean_3: float,
    roll_mean_5: float
) -> dict:
    """Executes prediction simulation for a given climate scenario."""
    model_payload = load_model()
    reg = model_payload["regressor"]
    clf = model_payload["classifier"]
    
    input_row = pd.DataFrame([{
        'Jan-Feb': jan_feb,
        'Mar-May': mar_may,
        'LAG_1': lag_1,
        'LAG_2': lag_2,
        'ROLL_MEAN_3': roll_mean_3,
        'ROLL_MEAN_5': roll_mean_5,
        'NORMAL_ANNUAL': normal_annual
    }])[FEATURE_COLS]
    
    pred_annual = float(reg.predict(input_row)[0])
    pred_cat = str(clf.predict(input_row)[0])
    probs = clf.predict_proba(input_row)[0]
    prob_dict = {c: round(float(p) * 100, 1) for c, p in zip(CLASSES, probs)}
    
    # Compute departure from normal
    dep_pct = ((pred_annual - normal_annual) / normal_annual) * 100 if normal_annual > 0 else 0.0
    
    return {
        "subdivision": subdivision,
        "predicted_annual_rainfall": round(pred_annual, 1),
        "normal_annual": round(normal_annual, 1),
        "predicted_departure_pct": round(dep_pct, 1),
        "predicted_category": pred_cat,
        "category_probabilities": prob_dict,
        "confidence": prob_dict.get(pred_cat, 0.0),
        "features_used": dict(input_row.iloc[0])
    }

if __name__ == "__main__":
    print("Training and saving rainfall_model.pkl...")
    res = train_and_save_model()
    print("Regression R2:", res['metrics']['regressor']['r2_score'])
    print("Classification Accuracy:", res['metrics']['classifier']['accuracy'])
