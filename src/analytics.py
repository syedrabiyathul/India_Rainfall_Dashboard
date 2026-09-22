"""
Analytics and statistical calculations engine for India Rainfall Dashboard.
All calculations operate strictly on actual dataset records.
"""

import pandas as pd
import numpy as np
from src.utils import MONTHS, SEASONS

def get_executive_kpis(df: pd.DataFrame) -> dict:
    """Computes high-level national summary KPIs."""
    nat_mean = float(df['ANNUAL'].mean())
    nat_max = float(df['ANNUAL'].max())
    nat_min = float(df['ANNUAL'].min())
    mon_mean = float(df['Jun-Sep'].mean())
    mon_share = float((mon_mean / nat_mean) * 100)
    
    # Yearly national averages
    yearly = df.groupby('YEAR')['ANNUAL'].mean()
    highest_year = int(yearly.idxmax())
    highest_val = float(yearly.max())
    lowest_year = int(yearly.idxmin())
    lowest_val = float(yearly.min())
    
    # Wettest and driest regions
    sub_means = df.groupby('SUBDIVISION')['ANNUAL'].mean()
    wettest_region = str(sub_means.idxmax())
    wettest_val = float(sub_means.max())
    driest_region = str(sub_means.idxmin())
    driest_val = float(sub_means.min())
    
    return {
        "total_records": int(len(df)),
        "time_period": f"{int(df['YEAR'].min())} – {int(df['YEAR'].max())}",
        "num_regions": int(df['SUBDIVISION'].nunique()),
        "average_rainfall": round(nat_mean, 1),
        "maximum_rainfall": round(nat_max, 1),
        "minimum_rainfall": round(nat_min, 1),
        "highest_rainfall_year": highest_year,
        "highest_rainfall_val": round(highest_val, 1),
        "lowest_rainfall_year": lowest_year,
        "lowest_rainfall_val": round(lowest_val, 1),
        "monsoon_share_pct": round(mon_share, 1),
        "wettest_region": wettest_region,
        "wettest_val": round(wettest_val, 1),
        "driest_region": driest_region,
        "driest_val": round(driest_val, 1)
    }

def get_annual_trend_data(df: pd.DataFrame, region: str = None, start_year: int = 1901, end_year: int = 2015, ma_window: int = 5) -> dict:
    """Calculates yearly trend, moving average, and linear regression slope."""
    df_filtered = df[(df['YEAR'] >= start_year) & (df['YEAR'] <= end_year)].copy()
    
    if region and region != "All-India":
        df_reg = df_filtered[df_filtered['SUBDIVISION'] == region].sort_values('YEAR').copy()
        normal_ann = float(df_reg['NORMAL_ANNUAL'].iloc[0]) if len(df_reg) > 0 else 0.0
    else:
        df_reg = df_filtered.groupby('YEAR')[['ANNUAL', 'Jun-Sep']].mean().reset_index()
        df_reg['SUBDIVISION'] = 'All-India'
        df_reg['NORMAL_ANNUAL'] = df.groupby('YEAR')['ANNUAL'].mean().mean()
        normal_ann = float(df_reg['NORMAL_ANNUAL'].iloc[0])
        
    df_reg['MA'] = df_reg['ANNUAL'].rolling(ma_window, min_periods=1).mean().round(1)
    
    # Linear slope
    x = df_reg['YEAR'].values
    y = df_reg['ANNUAL'].values
    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope, intercept = 0.0, float(y[0]) if len(y) > 0 else 0.0
    decadal_slope = round(float(slope * 10), 2)
    
    # Max and Min
    max_row = df_reg.loc[df_reg['ANNUAL'].idxmax()]
    min_row = df_reg.loc[df_reg['ANNUAL'].idxmin()]
    
    return {
        "data": df_reg,
        "slope": slope,
        "intercept": intercept,
        "decadal_slope": decadal_slope,
        "normal_annual": round(normal_ann, 1),
        "max_year": int(max_row['YEAR']),
        "max_val": round(float(max_row['ANNUAL']), 1),
        "min_year": int(min_row['YEAR']),
        "min_val": round(float(min_row['ANNUAL']), 1)
    }

def get_regional_summary_rankings(df: pd.DataFrame, start_year: int = 1901, end_year: int = 2015) -> pd.DataFrame:
    """Computes regional averages, min, max, std sorted by mean rainfall."""
    df_f = df[(df['YEAR'] >= start_year) & (df['YEAR'] <= end_year)]
    summary = df_f.groupby('SUBDIVISION')['ANNUAL'].agg(
        Mean='mean',
        Max='max',
        Min='min',
        Std='std',
        Observations='count'
    ).round(1).sort_values('Mean', ascending=False).reset_index()
    return summary

def get_departure_summary(df: pd.DataFrame, region: str) -> dict:
    """Analyzes departure percentages and IMD category frequency."""
    df_reg = df[df['SUBDIVISION'] == region].copy()
    if df_reg.empty:
        return {}
        
    cat_counts = df_reg['IMD_CATEGORY'].value_counts().to_dict()
    total = len(df_reg)
    excess_pct = round((cat_counts.get("Large Excess", 0) + cat_counts.get("Excess", 0)) / total * 100, 1)
    normal_pct = round(cat_counts.get("Normal", 0) / total * 100, 1)
    deficit_pct = round((cat_counts.get("Deficient", 0) + cat_counts.get("Scanty", 0)) / total * 100, 1)
    
    droughts = df_reg[df_reg['DEPARTURE_PCT'] <= -20.0][['YEAR', 'ANNUAL', 'NORMAL_ANNUAL', 'DEPARTURE_PCT']].sort_values('DEPARTURE_PCT')
    deluges = df_reg[df_reg['DEPARTURE_PCT'] >= 20.0][['YEAR', 'ANNUAL', 'NORMAL_ANNUAL', 'DEPARTURE_PCT']].sort_values('DEPARTURE_PCT', ascending=False)
    
    return {
        "category_counts": cat_counts,
        "excess_pct": excess_pct,
        "normal_pct": normal_pct,
        "deficit_pct": deficit_pct,
        "drought_years": droughts.reset_index(drop=True),
        "deluge_years": deluges.reset_index(drop=True)
    }
