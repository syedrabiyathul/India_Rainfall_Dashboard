"""
Data loading and preprocessing module for India Rainfall AI Analytics & Prediction Dashboard.
Handles 1901-2015 subdivision rainfall dataset and district-wise rainfall normal dataset.
"""

import os
import pandas as pd
import numpy as np

SUB_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rainfall in india 1901-2015.csv")
DIST_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "district wise rainfall normal.csv")

MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
SEASONS = ['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']

def classify_imd_category(dep_pct: float) -> str:
    """
    Standard India Meteorological Department (IMD) Rainfall Departure Categories:
    - Large Excess (LE): +60% or more
    - Excess (E): +20% to +59%
    - Normal (N): -19% to +19%
    - Deficient (D): -20% to -59%
    - Scanty / Large Deficient (LD): -60% or less
    """
    if pd.isna(dep_pct):
        return "Normal"
    if dep_pct >= 60.0:
        return "Large Excess"
    elif dep_pct >= 20.0:
        return "Excess"
    elif dep_pct >= -19.0:
        return "Normal"
    elif dep_pct >= -59.0:
        return "Deficient"
    else:
        return "Scanty"

def classify_simplified_category(dep_pct: float) -> str:
    """3-class category for robust ML classification."""
    if pd.isna(dep_pct):
        return "Normal"
    if dep_pct >= 20.0:
        return "Excess"
    elif dep_pct <= -20.0:
        return "Deficient"
    else:
        return "Normal"

def load_data():
    """
    Loads, cleans, and standardizes both historical and district normal datasets.
    Returns:
        df_sub (pd.DataFrame): 1901-2015 subdivision-level cleaned data with normals and departures
        df_dist (pd.DataFrame): District-level official normal rainfall benchmark data
    """
    # 1. Load historical subdivision rainfall
    df_sub = pd.read_csv(SUB_FILE)
    df_sub = df_sub.dropna(subset=['ANNUAL']).copy()
    
    # Impute missing monthly values using subdivision median
    for m in MONTHS:
        df_sub[m] = df_sub.groupby('SUBDIVISION')[m].transform(lambda x: x.fillna(x.median()))
        
    # Standardize seasonal calculations
    df_sub['Jan-Feb'] = df_sub['JAN'] + df_sub['FEB']
    df_sub['Mar-May'] = df_sub['MAR'] + df_sub['APR'] + df_sub['MAY']
    df_sub['Jun-Sep'] = df_sub['JUN'] + df_sub['JUL'] + df_sub['AUG'] + df_sub['SEP']
    df_sub['Oct-Dec'] = df_sub['OCT'] + df_sub['NOV'] + df_sub['DEC']
    df_sub['ANNUAL'] = df_sub['Jan-Feb'] + df_sub['Mar-May'] + df_sub['Jun-Sep'] + df_sub['Oct-Dec']
    
    # Calculate IMD Climatological Normal for each subdivision
    sub_normals = df_sub.groupby('SUBDIVISION')['ANNUAL'].transform('mean')
    df_sub['NORMAL_ANNUAL'] = sub_normals
    
    # Seasonal Normals
    for s in SEASONS:
        df_sub[f'NORMAL_{s}'] = df_sub.groupby('SUBDIVISION')[s].transform('mean')
        
    # Departure Percentages
    df_sub['DEPARTURE_PCT'] = ((df_sub['ANNUAL'] - df_sub['NORMAL_ANNUAL']) / df_sub['NORMAL_ANNUAL']) * 100
    df_sub['MONSOON_DEPARTURE_PCT'] = ((df_sub['Jun-Sep'] - df_sub['NORMAL_Jun-Sep']) / df_sub['NORMAL_Jun-Sep']) * 100
    
    # Meteorological Categories
    df_sub['IMD_CATEGORY'] = df_sub['DEPARTURE_PCT'].apply(classify_imd_category)
    df_sub['CATEGORY_3CLASS'] = df_sub['DEPARTURE_PCT'].apply(classify_simplified_category)
    
    # Monsoon Share % of Annual
    df_sub['MONSOON_SHARE_PCT'] = (df_sub['Jun-Sep'] / df_sub['ANNUAL']) * 100

    # 2. Load district rainfall normal
    df_dist = pd.read_csv(DIST_FILE)
    df_dist['STATE_UT_NAME'] = df_dist['STATE_UT_NAME'].str.strip()
    df_dist['DISTRICT'] = df_dist['DISTRICT'].str.strip()
    
    # Monsoon Share %
    df_dist['MONSOON_SHARE_PCT'] = (df_dist['Jun-Sep'] / df_dist['ANNUAL']) * 100
    
    return df_sub, df_dist

def get_subdivision_list(df_sub: pd.DataFrame) -> list:
    return sorted(df_sub['SUBDIVISION'].unique().tolist())

def get_state_list(df_dist: pd.DataFrame) -> list:
    return sorted(df_dist['STATE_UT_NAME'].unique().tolist())

def get_district_list(df_dist: pd.DataFrame, state: str = None) -> list:
    if state and state != "All States":
        return sorted(df_dist[df_dist['STATE_UT_NAME'] == state]['DISTRICT'].unique().tolist())
    return sorted(df_dist['DISTRICT'].unique().tolist())

def get_national_metrics(df_sub: pd.DataFrame):
    """Computes high-level executive overview statistics."""
    # Mean annual across all subdivisions and years
    national_mean = df_sub['ANNUAL'].mean()
    monsoon_mean = df_sub['Jun-Sep'].mean()
    monsoon_pct = (monsoon_mean / national_mean) * 100
    
    # Yearly national average
    yearly_nat = df_sub.groupby('YEAR')['ANNUAL'].mean()
    highest_year = int(yearly_nat.idxmax())
    highest_val = float(yearly_nat.max())
    lowest_year = int(yearly_nat.idxmin())
    lowest_val = float(yearly_nat.min())
    
    # Wettest and driest subdivisions
    sub_means = df_sub.groupby('SUBDIVISION')['ANNUAL'].mean().sort_values(ascending=False)
    wettest_sub = sub_means.index[0]
    wettest_val = sub_means.iloc[0]
    driest_sub = sub_means.index[-1]
    driest_val = sub_means.iloc[-1]
    
    return {
        "national_mean": round(national_mean, 1),
        "monsoon_mean": round(monsoon_mean, 1),
        "monsoon_share_pct": round(monsoon_pct, 1),
        "highest_year": highest_year,
        "highest_val": round(highest_val, 1),
        "lowest_year": lowest_year,
        "lowest_val": round(lowest_val, 1),
        "wettest_sub": wettest_sub,
        "wettest_val": round(wettest_val, 1),
        "driest_sub": driest_sub,
        "driest_val": round(driest_val, 1),
        "total_records": len(df_sub),
        "year_span": f"{df_sub['YEAR'].min()} - {df_sub['YEAR'].max()}"
    }

if __name__ == "__main__":
    df_s, df_d = load_data()
    print(f"Loaded df_sub: {df_s.shape}, df_dist: {df_d.shape}")
    metrics = get_national_metrics(df_s)
    print("Metrics:", metrics)
