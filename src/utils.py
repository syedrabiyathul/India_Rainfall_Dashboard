"""
Utility functions and constants for the India Rainfall AI Dashboard.
"""

import os
import pandas as pd
import numpy as np

# Project Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

RAINFALL_CSV = os.path.join(DATA_DIR, "rainfall.csv")
DISTRICT_NORMAL_CSV = os.path.join(DATA_DIR, "district_normal.csv")
MODEL_PKL = os.path.join(MODELS_DIR, "rainfall_model.pkl")

MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
SEASONS = ['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']
FEATURE_COLS = ['Jan-Feb', 'Mar-May', 'LAG_1', 'LAG_2', 'ROLL_MEAN_3', 'ROLL_MEAN_5', 'NORMAL_ANNUAL']
CLASSES = ['Deficient', 'Normal', 'Excess']

def classify_imd(departure_pct: float) -> str:
    """
    Standard IMD departure categorization:
    - Large Excess: >= +60%
    - Excess: +20% to +59.9%
    - Normal: -19.9% to +19.9%
    - Deficient: -20% to -59.9%
    - Scanty / Large Deficient: <= -60%
    """
    if pd.isna(departure_pct):
        return "Normal"
    if departure_pct >= 60.0:
        return "Large Excess"
    elif departure_pct >= 20.0:
        return "Excess"
    elif departure_pct >= -19.0:
        return "Normal"
    elif departure_pct >= -59.0:
        return "Deficient"
    else:
        return "Scanty"

def classify_simplified(departure_pct: float) -> str:
    """3-Class mapping for ML models (Deficient, Normal, Excess)."""
    if pd.isna(departure_pct):
        return "Normal"
    if departure_pct >= 20.0:
        return "Excess"
    elif departure_pct <= -20.0:
        return "Deficient"
    else:
        return "Normal"
