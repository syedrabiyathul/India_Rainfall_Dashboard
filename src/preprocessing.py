"""
Data preprocessing, cleaning, normal derivation, and validation module.
"""

import os
import pandas as pd
import numpy as np
from src.utils import RAINFALL_CSV, DISTRICT_NORMAL_CSV, MONTHS, SEASONS, classify_imd, classify_simplified

def inspect_raw_dataset():
    """
    Performs comprehensive diagnostic inspection of raw rainfall datasets.
    Returns dictionary with all findings for reporting.
    """
    df_raw = pd.read_csv(RAINFALL_CSV)
    df_dist_raw = pd.read_csv(DISTRICT_NORMAL_CSV) if os.path.exists(DISTRICT_NORMAL_CSV) else None
    
    # 1. Dimensions & Columns
    rows, cols = df_raw.shape
    columns = list(df_raw.columns)
    dtypes = {c: str(t) for c, t in df_raw.dtypes.items()}
    
    # 2. Time Horizon & Regions
    time_min = int(df_raw['YEAR'].min())
    time_max = int(df_raw['YEAR'].max())
    time_range = f"{time_min} - {time_max}"
    unique_subdivisions = sorted(df_raw['SUBDIVISION'].dropna().unique().tolist())
    num_subdivisions = len(unique_subdivisions)
    
    # 3. Missing Values & Duplicates
    missing_counts = df_raw.isnull().sum().to_dict()
    total_missing = int(df_raw.isnull().sum().sum())
    rows_with_null = int(df_raw.isnull().any(axis=1).sum())
    duplicate_rows = int(df_raw.duplicated().sum())
    
    # 4. Outlier Analysis using IQR on ANNUAL
    q1 = float(df_raw['ANNUAL'].quantile(0.25))
    q3 = float(df_raw['ANNUAL'].quantile(0.75))
    iqr = q3 - q1
    lower_bound = max(0.0, q1 - 1.5 * iqr)
    upper_bound = q3 + 1.5 * iqr
    outliers = df_raw[(df_raw['ANNUAL'] < lower_bound) | (df_raw['ANNUAL'] > upper_bound)]
    num_outliers = int(len(outliers))
    
    # 5. District Dataset Info if present
    dist_info = {}
    if df_dist_raw is not None:
        dist_info = {
            "rows": int(len(df_dist_raw)),
            "columns": list(df_dist_raw.columns),
            "states_count": int(df_dist_raw['STATE_UT_NAME'].nunique()),
            "districts_count": int(df_dist_raw['DISTRICT'].nunique()),
            "states": sorted(df_dist_raw['STATE_UT_NAME'].unique().tolist()),
            "missing_values": int(df_dist_raw.isnull().sum().sum())
        }

    return {
        "rows": rows,
        "cols": cols,
        "columns": columns,
        "dtypes": dtypes,
        "time_min": time_min,
        "time_max": time_max,
        "time_range": time_range,
        "num_subdivisions": num_subdivisions,
        "subdivisions": unique_subdivisions,
        "missing_counts": missing_counts,
        "total_missing": total_missing,
        "rows_with_null": rows_with_null,
        "duplicate_rows": duplicate_rows,
        "annual_mean": round(float(df_raw['ANNUAL'].dropna().mean()), 2),
        "annual_min": round(float(df_raw['ANNUAL'].dropna().min()), 2),
        "annual_max": round(float(df_raw['ANNUAL'].dropna().max()), 2),
        "annual_std": round(float(df_raw['ANNUAL'].dropna().std()), 2),
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "num_outliers": num_outliers,
        "district_dataset": dist_info
    }

def load_clean_data():
    """
    Loads raw CSV, performs missing-value imputation with subdivision median,
    re-computes consistent seasonal and annual sums, computes long-term normals,
    calculates departure percentages, and assigns IMD categories.
    """
    df = pd.read_csv(RAINFALL_CSV)
    df = df.dropna(subset=['ANNUAL']).copy()
    
    # Impute missing monthly values using subdivision median
    for m in MONTHS:
        df[m] = df.groupby('SUBDIVISION')[m].transform(lambda x: x.fillna(x.median()))
        
    # Recompute seasonal columns to guarantee consistency
    df['Jan-Feb'] = df['JAN'] + df['FEB']
    df['Mar-May'] = df['MAR'] + df['APR'] + df['MAY']
    df['Jun-Sep'] = df['JUN'] + df['JUL'] + df['AUG'] + df['SEP']
    df['Oct-Dec'] = df['OCT'] + df['NOV'] + df['DEC']
    df['ANNUAL'] = df['Jan-Feb'] + df['Mar-May'] + df['Jun-Sep'] + df['Oct-Dec']
    
    # Derive subdivision climatological normals
    df['NORMAL_ANNUAL'] = df.groupby('SUBDIVISION')['ANNUAL'].transform('mean')
    for s in SEASONS:
        df[f'NORMAL_{s}'] = df.groupby('SUBDIVISION')[s].transform('mean')
        
    # Departure calculations
    df['DEPARTURE_PCT'] = ((df['ANNUAL'] - df['NORMAL_ANNUAL']) / df['NORMAL_ANNUAL']) * 100
    df['MONSOON_DEPARTURE_PCT'] = ((df['Jun-Sep'] - df['NORMAL_Jun-Sep']) / df['NORMAL_Jun-Sep']) * 100
    
    # IMD meteorological categories
    df['IMD_CATEGORY'] = df['DEPARTURE_PCT'].apply(classify_imd)
    df['CATEGORY_3CLASS'] = df['DEPARTURE_PCT'].apply(classify_simplified)
    
    # Monsoon Share %
    df['MONSOON_SHARE_PCT'] = (df['Jun-Sep'] / df['ANNUAL']) * 100
    
    # Load district normals if available
    df_dist = None
    if os.path.exists(DISTRICT_NORMAL_CSV):
        df_dist = pd.read_csv(DISTRICT_NORMAL_CSV)
        df_dist['STATE_UT_NAME'] = df_dist['STATE_UT_NAME'].str.strip()
        df_dist['DISTRICT'] = df_dist['DISTRICT'].str.strip()
        df_dist['MONSOON_SHARE_PCT'] = (df_dist['Jun-Sep'] / df_dist['ANNUAL']) * 100
        
    return df, df_dist

if __name__ == "__main__":
    report = inspect_raw_dataset()
    print("Inspection complete. Total rows:", report['rows'], "Subdivisions:", report['num_subdivisions'])
