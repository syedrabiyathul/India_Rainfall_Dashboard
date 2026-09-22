"""
Automated AI Data Insights Engine for India Rainfall Analytics.
Generates statistical anomaly detection, decadal trend shifts,
monsoon dependency indices, and historical climate intelligence cards.
"""

import pandas as pd
import numpy as np

def generate_executive_insights(df_sub: pd.DataFrame) -> list:
    """Generates structured statistical insight cards for the executive dashboard."""
    insights = []
    
    # 1. National Decadal Shift Analysis
    df_copy = df_sub.copy()
    df_copy['DECADE'] = (df_copy['YEAR'] // 10) * 10
    decade_stats = df_copy.groupby('DECADE')['ANNUAL'].mean()
    
    earliest_decade_val = round(decade_stats.iloc[0], 1)
    latest_decade_val = round(decade_stats.iloc[-1], 1)
    dec_diff = round(latest_decade_val - earliest_decade_val, 1)
    dec_diff_pct = round((dec_diff / earliest_decade_val) * 100, 1)
    
    insights.append({
        "id": "decadal_trend",
        "title": "Decadal Climate Precipitation Shift",
        "category": "Climate Trend",
        "badge": "Trend Watch" if abs(dec_diff_pct) > 5 else "Stable",
        "severity": "info" if abs(dec_diff_pct) <= 5 else "warning",
        "metric_label": "1900s vs 2010s Change",
        "metric_value": f"{dec_diff:+} mm ({dec_diff_pct:+}%)",
        "summary": (
            f"National average annual rainfall in the earliest recorded decade (1900s) was {earliest_decade_val} mm, "
            f"compared to {latest_decade_val} mm in recent years (2010-2015). "
            f"This represents a net shift of {dec_diff:+} mm ({dec_diff_pct:+}%)."
        )
    })
    
    # 2. Monsoon Concentration & Asymmetry
    nat_annual = df_sub['ANNUAL'].mean()
    nat_monsoon = df_sub['Jun-Sep'].mean()
    monsoon_pct = round((nat_monsoon / nat_annual) * 100, 1)
    
    # Subdivision with highest and lowest monsoon dependency
    sub_monsoon_share = (df_sub.groupby('SUBDIVISION')['Jun-Sep'].sum() / df_sub.groupby('SUBDIVISION')['ANNUAL'].sum() * 100).round(1)
    highest_mon_sub = sub_monsoon_share.idxmax()
    highest_mon_val = sub_monsoon_share.max()
    lowest_mon_sub = sub_monsoon_share.idxmin()
    lowest_mon_val = sub_monsoon_share.min()
    
    insights.append({
        "id": "monsoon_dependency",
        "title": "Southwest Monsoon Dominance & Asymmetry",
        "category": "Seasonal Profile",
        "badge": "Monsoon Heavy",
        "severity": "success",
        "metric_label": "All-India Monsoon Share",
        "metric_value": f"{monsoon_pct}% of Annual",
        "summary": (
            f"Across the 115-year record, {monsoon_pct}% of India's annual rainfall is concentrated in the 4-month Southwest Monsoon (Jun-Sep). "
            f"Maximum dependency occurs in {highest_mon_sub} ({highest_mon_val}% of rainfall), "
            f"while {lowest_mon_sub} receives only {lowest_mon_val}% during Jun-Sep due to major Northeast Monsoon (Oct-Dec) contribution."
        )
    })
    
    # 3. Severe National Drought Identification
    nat_yearly = df_sub.groupby('YEAR')['ANNUAL'].mean()
    nat_longterm_mean = nat_yearly.mean()
    yearly_dep = ((nat_yearly - nat_longterm_mean) / nat_longterm_mean * 100).round(1)
    
    worst_drought_year = int(yearly_dep.idxmin())
    worst_drought_val = round(nat_yearly.loc[worst_drought_year], 1)
    worst_drought_dep = round(yearly_dep.loc[worst_drought_year], 1)
    
    insights.append({
        "id": "worst_drought",
        "title": "Centennial Record Drought Benchmark",
        "category": "Extreme Events",
        "badge": "Deficit Anomaly",
        "severity": "danger",
        "metric_label": f"Worst Drought Year ({worst_drought_year})",
        "metric_value": f"{worst_drought_dep}% Departure",
        "summary": (
            f"The driest year in the modern meteorological archive occurred in {worst_drought_year}, "
            f"with an all-India average of just {worst_drought_val} mm ({worst_drought_dep}% departure from normal). "
            f"Other catastrophic benchmark drought years include 1918, 1965, 1979, 2002, and 2009."
        )
    })
    
    # 4. Centennial Peak Rainfall Benchmark
    peak_wet_year = int(yearly_dep.idxmax())
    peak_wet_val = round(nat_yearly.loc[peak_wet_year], 1)
    peak_wet_dep = round(yearly_dep.loc[peak_wet_year], 1)
    
    insights.append({
        "id": "peak_deluge",
        "title": "Centennial Record Deluge Benchmark",
        "category": "Extreme Events",
        "badge": "Excess Anomaly",
        "severity": "success",
        "metric_label": f"Peak Deluge Year ({peak_wet_year})",
        "metric_value": f"+{peak_wet_dep}% Departure",
        "summary": (
            f"The wettest year nationally was recorded in {peak_wet_year}, "
            f"reaching {peak_wet_val} mm (+{peak_wet_dep}% above normal). "
            f"Widespread pan-Indian flooding occurred during this active monsoon sequence."
        )
    })
    
    # 5. Volatility & Coefficient of Variation
    sub_cv = (df_sub.groupby('SUBDIVISION')['ANNUAL'].std() / df_sub.groupby('SUBDIVISION')['ANNUAL'].mean() * 100).round(1)
    most_volatile_sub = sub_cv.idxmax()
    most_volatile_val = sub_cv.max()
    most_stable_sub = sub_cv.idxmin()
    most_stable_val = sub_cv.min()
    
    insights.append({
        "id": "volatility_analysis",
        "title": "Inter-Annual Rainfall Volatility Index",
        "category": "Risk Analysis",
        "badge": "High Variance",
        "severity": "warning",
        "metric_label": f"Peak Volatility ({most_volatile_sub})",
        "metric_value": f"{most_volatile_val}% CV",
        "summary": (
            f"{most_volatile_sub} exhibits the highest rainfall unpredictability with a Coefficient of Variation (CV) of {most_volatile_val}%. "
            f"Conversely, {most_stable_sub} demonstrates the most consistent climatological regime with a CV of only {most_stable_val}%."
        )
    })
    
    return insights

def get_subdivision_deep_dive(df_sub: pd.DataFrame, subdivision: str) -> dict:
    """Computes targeted analytics for a specific subdivision."""
    df_reg = df_sub[df_sub['SUBDIVISION'] == subdivision].sort_values('YEAR').copy()
    if df_reg.empty:
        return {}
        
    normal_ann = round(df_reg['NORMAL_ANNUAL'].iloc[0], 1)
    normal_mon = round(df_reg['NORMAL_Jun-Sep'].iloc[0], 1)
    
    # Category counts
    cat_counts = df_reg['IMD_CATEGORY'].value_counts().to_dict()
    total_years = len(df_reg)
    excess_pct = round((cat_counts.get("Large Excess", 0) + cat_counts.get("Excess", 0)) / total_years * 100, 1)
    normal_pct = round(cat_counts.get("Normal", 0) / total_years * 100, 1)
    deficit_pct = round((cat_counts.get("Deficient", 0) + cat_counts.get("Scanty", 0)) / total_years * 100, 1)
    
    # Extremes
    wettest_row = df_reg.loc[df_reg['ANNUAL'].idxmax()]
    driest_row = df_reg.loc[df_reg['ANNUAL'].idxmin()]
    
    # Linear slope
    x = df_reg['YEAR'].values
    y = df_reg['ANNUAL'].values
    slope, intercept = np.polyfit(x, y, 1)
    decadal_rate = round(slope * 10, 2)
    
    return {
        "subdivision": subdivision,
        "normal_annual": normal_ann,
        "normal_monsoon": normal_mon,
        "total_years_analyzed": total_years,
        "excess_years_pct": excess_pct,
        "normal_years_pct": normal_pct,
        "deficit_years_pct": deficit_pct,
        "wettest_year": int(wettest_row['YEAR']),
        "wettest_val": round(wettest_row['ANNUAL'], 1),
        "driest_year": int(driest_row['YEAR']),
        "driest_val": round(driest_row['ANNUAL'], 1),
        "decadal_trend_rate_mm": decadal_rate
    }
