"""
Visualization module providing modern Plotly charts with curated dark aesthetics.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from src.utils import MONTHS, CLASSES

THEME = "plotly_dark"

def plot_all_india_timeline(nat_df: pd.DataFrame, normal_val: float) -> go.Figure:
    """Generates the All-India Annual Rainfall timeline with 5Y rolling average."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=nat_df['YEAR'], y=nat_df['ANNUAL'],
        name="Annual Rainfall", marker_color='rgba(56, 189, 248, 0.45)',
        hovertemplate="Year: %{x}<br>Rainfall: %{y:.1f} mm<extra></extra>"
    ))
    if '5Y_MA' in nat_df.columns:
        fig.add_trace(go.Scatter(
            x=nat_df['YEAR'], y=nat_df['5Y_MA'],
            name="5-Yr Rolling Avg", line=dict(color='#38bdf8', width=2.5),
            hovertemplate="Year: %{x}<br>5Y Moving Avg: %{y:.1f} mm<extra></extra>"
        ))
    fig.add_hline(
        y=normal_val, line_dash="dash", line_color="#f59e0b",
        annotation_text=f"Normal: {normal_val:.1f} mm", annotation_position="top left"
    )
    fig.update_layout(
        title="All-India Annual Rainfall Dynamics (1901–2015)",
        xaxis_title="Year", yaxis_title="Rainfall (mm)",
        template=THEME, height=420, legend=dict(orientation="h", y=1.1)
    )
    return fig

def plot_seasonal_donut(season_values: list) -> go.Figure:
    """Generates a donut chart of seasonal precipitation shares."""
    labels = ['Winter (Jan-Feb)', 'Pre-Monsoon (Mar-May)', 'Monsoon (Jun-Sep)', 'Post-Monsoon (Oct-Dec)']
    fig = px.pie(
        names=labels, values=season_values,
        title="All-India Seasonal Precipitation Breakdown",
        hole=0.55,
        color_discrete_sequence=['#94a3b8', '#34d399', '#38bdf8', '#818cf8']
    )
    fig.update_layout(template=THEME, height=420)
    return fig

def plot_multi_region_history(df_hist: pd.DataFrame) -> go.Figure:
    """Generates a multi-line comparison for selected subdivisions."""
    fig = px.line(
        df_hist, x='YEAR', y='ANNUAL', color='SUBDIVISION', markers=True,
        title="Comparative Historical Rainfall Timeline",
        labels={'ANNUAL': 'Annual Rainfall (mm)', 'YEAR': 'Year'},
        color_discrete_sequence=['#38bdf8', '#f43f5e', '#10b981', '#fbbf24', '#818cf8', '#a78bfa']
    )
    fig.update_layout(template=THEME, height=450, hovermode="x unified")
    return fig

def plot_annual_trend(trend_dict: dict, region_name: str, ma_window: int) -> go.Figure:
    """Generates annual trend chart with linear regression and moving average."""
    df_reg = trend_dict['data']
    slope = trend_dict['slope']
    intercept = trend_dict['intercept']
    dec_slope = trend_dict['decadal_slope']
    normal_val = trend_dict['normal_annual']
    
    x = df_reg['YEAR'].values
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_reg['YEAR'], y=df_reg['ANNUAL'],
        name="Annual Rainfall", marker_color='rgba(56, 189, 248, 0.4)'
    ))
    fig.add_trace(go.Scatter(
        x=df_reg['YEAR'], y=df_reg['MA'],
        name=f"{ma_window}-Year Moving Average", line=dict(color='#38bdf8', width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=df_reg['YEAR'], y=slope * x + intercept,
        name=f"Linear Trend ({dec_slope:+} mm/decade)",
        line=dict(color='#f43f5e', width=2, dash='dash')
    ))
    fig.add_hline(y=normal_val, line_dash="dot", line_color="#fbbf24", annotation_text="Normal")
    fig.update_layout(
        title=f"Annual Trend & Regression: {region_name}",
        xaxis_title="Year", yaxis_title="Rainfall (mm)",
        template=THEME, height=420, legend=dict(orientation="h", y=1.1)
    )
    return fig

def plot_departure_bars(df_reg: pd.DataFrame) -> go.Figure:
    """Generates departure % bar chart with blue for positive, red for negative."""
    df_reg = df_reg.copy()
    df_reg['COLOR'] = np.where(df_reg['DEPARTURE_PCT'] >= 0, '#38bdf8', '#f43f5e')
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_reg['YEAR'], y=df_reg['DEPARTURE_PCT'],
        marker_color=df_reg['COLOR'], name="Departure %"
    ))
    fig.add_hline(y=20, line_dash="dash", line_color="#34d399", annotation_text="+20% Excess")
    fig.add_hline(y=-20, line_dash="dash", line_color="#fb7185", annotation_text="-20% Deficient")
    fig.update_layout(
        title="Annual Departure % from Climatological Normal",
        xaxis_title="Year", yaxis_title="Departure (%)",
        template=THEME, height=350
    )
    return fig

def plot_monthly_boxplots(df_sub_months: pd.DataFrame) -> go.Figure:
    """Generates monthly rainfall variability boxplots across 12 months."""
    df_melt = pd.melt(
        df_sub_months, id_vars=['YEAR'], value_vars=MONTHS,
        var_name='Month', value_name='Rainfall'
    )
    fig = px.box(
        df_melt, x='Month', y='Rainfall', color='Month',
        title="Monthly Rainfall Distribution (12 Months)",
        labels={'Rainfall': 'Monthly Rainfall (mm)'},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_layout(template=THEME, height=420, showlegend=False)
    return fig

def plot_month_year_heatmap(df_sub_months: pd.DataFrame, region_name: str) -> go.Figure:
    """Generates 2D Month vs Year precipitation heatmap."""
    heatmap_data = df_sub_months.set_index('YEAR')[MONTHS].T
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Year", y="Month", color="Rainfall (mm)"),
        color_continuous_scale="Blues", aspect="auto",
        title=f"Yearly-Monthly Rainfall Matrix: {region_name}"
    )
    fig.update_layout(template=THEME, height=380)
    return fig

def plot_confusion_matrix(cm_matrix: list, classes: list = CLASSES) -> go.Figure:
    """Generates confusion matrix heatmap."""
    fig = px.imshow(
        cm_matrix, x=classes, y=classes,
        color_continuous_scale='Blues', text_auto=True,
        title="Confusion Matrix (Predicted vs Actual)",
        labels=dict(x="Predicted Category", y="True Category", color="Count")
    )
    fig.update_layout(template=THEME, height=380)
    return fig

def plot_feature_importance(feat_dict: dict) -> go.Figure:
    """Generates horizontal bar chart of feature importances."""
    df_feat = pd.DataFrame(list(feat_dict.items()), columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)
    fig = px.bar(
        df_feat, x='Importance', y='Feature', orientation='h',
        title="Model Feature Importance",
        labels={'Importance': 'Relative Importance', 'Feature': 'Feature'},
        color='Importance', color_continuous_scale='Tealgrn'
    )
    fig.update_layout(template=THEME, height=380)
    return fig
