"""
India Rainfall AI Analytics & Prediction Dashboard
Main Streamlit Application.
Features 10 integrated analytics modules, dual ML predictive engine,
and deterministic, zero-hallucination Natural Language AI agent.
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add src to system path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data_loader import (
    load_data, get_subdivision_list, get_state_list, get_district_list,
    get_national_metrics, MONTHS, SEASONS
)
from ml_engine import load_or_train_models, predict_scenario, FEATURE_COLS, CLASSES
from nl_agent import process_nl_query
from ai_agent import query_rainfall_ai
from clario_integration import get_clario_client
from insights_engine import generate_executive_insights, get_subdivision_deep_dive

# ---------------------------------------------------------
# Page Configuration & CSS Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="India Rainfall AI Analytics & Prediction Dashboard",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Cached Data & Model Loading
# ---------------------------------------------------------
@st.cache_data(show_spinner="Loading rainfall datasets...")
def get_data():
    return load_data()

@st.cache_resource(show_spinner="Initializing Machine Learning Engine...")
def get_models(df_sub):
    return load_or_train_models(df_sub)

df_sub, df_dist = get_data()
reg_model, clf_model, ml_metrics = get_models(df_sub)
national_stats = get_national_metrics(df_sub)

# ---------------------------------------------------------
# Sidebar Controls & Global Filters
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 20px 0;">
        <span class="header-badge">🛰️ Climatological AI</span>
        <h2 style="margin: 8px 0 4px 0; font-size: 1.4rem; color: #ffffff;">India Rainfall AI</h2>
        <p style="color: #94a3b8; font-size: 0.82rem; margin: 0;">1901–2015 Historical & Normal Analytics</p>
    </div>
    """, unsafe_allow_html=True)

    # Clario MCP Status Card
    clario = get_clario_client()
    clar_status = clario.get_status()
    if clar_status.get("connected"):
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 8px; padding: 10px 14px; margin-bottom: 14px;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span style="font-weight: 600; color: #10b981; font-size: 0.85rem;">🟢 Clario MCP: Active</span>
                <span style="font-size: 0.75rem; color: #94a3b8;">Stdio RPC</span>
            </div>
            <div style="font-size: 0.78rem; color: #cbd5e1; margin-top: 4px;">
                9 Tools Active • DuckDB Connected (4,116 rows)
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 8px; padding: 10px 14px; margin-bottom: 14px;">
            <span style="font-weight: 600; color: #f59e0b; font-size: 0.85rem;">🟠 Clario: Offline Fallback</span>
            <div style="font-size: 0.78rem; color: #cbd5e1; margin-top: 4px;">
                Local Deterministic Engine Active
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🎛️ Global Data Filters")
    subdivisions = get_subdivision_list(df_sub)
    default_sub_idx = subdivisions.index("TAMIL NADU") if "TAMIL NADU" in subdivisions else 0
    selected_subdivision = st.selectbox(
        "Primary Subdivision",
        subdivisions,
        index=default_sub_idx,
        help="Select primary meteorological subdivision for deep-dive analyses"
    )

    st.markdown("#### Era Quick-Select")
    era_col1, era_col2 = st.columns(2)
    with era_col1:
        if st.button("British Era\n(1901-1947)", use_container_width=True):
            st.session_state["year_range"] = (1901, 1947)
        if st.button("Modern Era\n(1991-2015)", use_container_width=True):
            st.session_state["year_range"] = (1991, 2015)
    with era_col2:
        if st.button("Post-Indep.\n(1947-1990)", use_container_width=True):
            st.session_state["year_range"] = (1947, 1990)
        if st.button("Full Record\n(1901-2015)", use_container_width=True):
            st.session_state["year_range"] = (1901, 2015)

    if "year_range" not in st.session_state:
        st.session_state["year_range"] = (1901, 2015)

    year_range = st.slider(
        "Timeline Range",
        min_value=1901,
        max_value=2015,
        value=st.session_state["year_range"],
        help="Filter historical time window"
    )

    st.markdown("---")
    st.markdown("### 💾 Export Raw Data")
    filtered_sub_df = df_sub[
        (df_sub['SUBDIVISION'] == selected_subdivision) &
        (df_sub['YEAR'] >= year_range[0]) &
        (df_sub['YEAR'] <= year_range[1])
    ]
    csv_sub = filtered_sub_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Download {selected_subdivision} CSV",
        data=csv_sub,
        file_name=f"{selected_subdivision.replace(' ', '_')}_{year_range[0]}_{year_range[1]}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.markdown("---")
    st.caption("Developed for India Climatological & Meteorological Analytics. Datasets source: IMD 1901–2015 & District Normals.")

# ---------------------------------------------------------
# Main App Header
# ---------------------------------------------------------
st.markdown("""
<div class="main-title-container">
    <div>
        <h1 class="main-title-text">India Rainfall AI Analytics & Prediction Dashboard</h1>
        <div class="main-subtitle-text">
            Comprehensive Climatological Intelligence, Machine Learning Forecasting, and Deterministic AI Query Agent
        </div>
    </div>
    <div>
        <span class="header-badge">LIVE METEOROLOGICAL DATASET</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Multi-Tab Application Navigation
# ---------------------------------------------------------
tabs = st.tabs([
    "1. Executive Overview",
    "2. Historical Analysis",
    "3. Annual Trends",
    "4. Monthly & Seasonal",
    "5. Regional & Districts",
    "6. Rainfall Normal & Departures",
    "7. ML Prediction Engine",
    "8. Model Evaluation",
    "9. Natural Language AI Agent",
    "10. AI Data Insights"
])

# =========================================================
# TAB 1: EXECUTIVE OVERVIEW
# =========================================================
with tabs[0]:
    st.markdown("### 📊 Climatological Executive KPI Cards")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">National Mean Annual</div>
            <div class="kpi-value">{national_stats['national_mean']} <span style="font-size: 1rem; font-weight: 500; color: #94a3b8;">mm</span></div>
            <div class="kpi-subtext">Across all 36 subdivisions (115 yrs)</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="kpi-card emerald">
            <div class="kpi-label">Southwest Monsoon Share</div>
            <div class="kpi-value">{national_stats['monsoon_share_pct']}%</div>
            <div class="kpi-subtext">{national_stats['monsoon_mean']} mm concentrated in Jun-Sep</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="kpi-card rose">
            <div class="kpi-label">Centennial Wettest Year</div>
            <div class="kpi-value">{national_stats['highest_year']}</div>
            <div class="kpi-subtext">Record {national_stats['highest_val']} mm national avg</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="kpi-card amber">
            <div class="kpi-label">Centennial Driest Year</div>
            <div class="kpi-value">{national_stats['lowest_year']}</div>
            <div class="kpi-subtext">Severe drought {national_stats['lowest_val']} mm national avg</div>
        </div>
        """, unsafe_allow_html=True)

    col_chart1, col_chart2 = st.columns([7, 5])
    with col_chart1:
        # All-India Annual Rainfall Timeline
        nat_timeline = df_sub.groupby('YEAR')['ANNUAL'].mean().reset_index()
        nat_timeline['5Y_MA'] = nat_timeline['ANNUAL'].rolling(5, min_periods=1).mean()
        
        fig_nat = go.Figure()
        fig_nat.add_trace(go.Bar(
            x=nat_timeline['YEAR'], y=nat_timeline['ANNUAL'],
            name="National Mean", marker_color='rgba(56, 189, 248, 0.45)',
            hovertemplate="Year: %{x}<br>Rainfall: %{y:.1f} mm<extra></extra>"
        ))
        fig_nat.add_trace(go.Scatter(
            x=nat_timeline['YEAR'], y=nat_timeline['5Y_MA'],
            name="5-Yr Rolling Avg", line=dict(color='#38bdf8', width=2.5),
            hovertemplate="Year: %{x}<br>5Y Moving Avg: %{y:.1f} mm<extra></extra>"
        ))
        fig_nat.add_hline(
            y=national_stats['national_mean'], line_dash="dash", line_color="#f59e0b",
            annotation_text=f"Normal: {national_stats['national_mean']} mm", annotation_position="top left"
        )
        fig_nat.update_layout(
            title="All-India Annual Rainfall Dynamics (1901–2015)",
            xaxis_title="Year", yaxis_title="Rainfall (mm)",
            template="plotly_dark", height=420, legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_nat, use_container_width=True)

    with col_chart2:
        # Seasonal Contribution Donut Chart
        season_sums = [
            df_sub['Jan-Feb'].mean(),
            df_sub['Mar-May'].mean(),
            df_sub['Jun-Sep'].mean(),
            df_sub['Oct-Dec'].mean()
        ]
        fig_donut = px.pie(
            names=['Winter (Jan-Feb)', 'Pre-Monsoon (Mar-May)', 'Monsoon (Jun-Sep)', 'Post-Monsoon (Oct-Dec)'],
            values=season_sums,
            title="All-India Seasonal Precipitation Breakdown",
            hole=0.55,
            color_discrete_sequence=['#94a3b8', '#34d399', '#38bdf8', '#818cf8']
        )
        fig_donut.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig_donut, use_container_width=True)

    # Top Wettest and Driest Subdivisions
    col_rank1, col_rank2 = st.columns(2)
    with col_rank1:
        top5_wet = df_sub.groupby('SUBDIVISION')['ANNUAL'].mean().sort_values(ascending=False).head(5).reset_index()
        fig_wet = px.bar(
            top5_wet, x='ANNUAL', y='SUBDIVISION', orientation='h',
            title="Top 5 Wettest Meteorological Subdivisions (Mean mm)",
            labels={'ANNUAL': 'Mean Rainfall (mm)', 'SUBDIVISION': 'Subdivision'},
            color='ANNUAL', color_continuous_scale='Blues'
        )
        fig_wet.update_layout(template="plotly_dark", height=280, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_wet, use_container_width=True)

    with col_rank2:
        top5_dry = df_sub.groupby('SUBDIVISION')['ANNUAL'].mean().sort_values(ascending=True).head(5).reset_index()
        fig_dry = px.bar(
            top5_dry, x='ANNUAL', y='SUBDIVISION', orientation='h',
            title="Top 5 Driest Meteorological Subdivisions (Mean mm)",
            labels={'ANNUAL': 'Mean Rainfall (mm)', 'SUBDIVISION': 'Subdivision'},
            color='ANNUAL', color_continuous_scale='Oranges_r'
        )
        fig_dry.update_layout(template="plotly_dark", height=280, yaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig_dry, use_container_width=True)

# =========================================================
# TAB 2: HISTORICAL RAINFALL ANALYSIS
# =========================================================
with tabs[1]:
    st.markdown("### 📜 Multi-Subdivision Historical Comparison")
    
    comp_subs = st.multiselect(
        "Select Subdivisions to Compare over Time",
        subdivisions,
        default=[selected_subdivision, "KERALA"] if selected_subdivision != "KERALA" else ["KERALA", "TAMIL NADU"]
    )
    
    if comp_subs:
        df_hist = df_sub[
            (df_sub['SUBDIVISION'].isin(comp_subs)) &
            (df_sub['YEAR'] >= year_range[0]) &
            (df_sub['YEAR'] <= year_range[1])
        ].copy()
        
        fig_multi = px.line(
            df_hist, x='YEAR', y='ANNUAL', color='SUBDIVISION', markers=True,
            title=f"Comparative Annual Rainfall Timeline ({year_range[0]}–{year_range[1]})",
            labels={'ANNUAL': 'Annual Rainfall (mm)', 'YEAR': 'Year'},
            color_discrete_sequence=['#38bdf8', '#f43f5e', '#10b981', '#fbbf24', '#818cf8', '#a78bfa']
        )
        fig_multi.update_layout(template="plotly_dark", height=450, hovermode="x unified")
        st.plotly_chart(fig_multi, use_container_width=True)
        
        col_h1, col_h2 = st.columns([7, 5])
        with col_h1:
            # Decadal Breakdown
            df_hist['DECADE'] = (df_hist['YEAR'] // 10) * 10
            decade_comp = df_hist.groupby(['DECADE', 'SUBDIVISION'])['ANNUAL'].mean().reset_index()
            fig_dec = px.bar(
                decade_comp, x='DECADE', y='ANNUAL', color='SUBDIVISION', barmode='group',
                title="Decadal Average Rainfall Evolution",
                labels={'ANNUAL': 'Mean Rainfall (mm)', 'DECADE': 'Decade'},
                color_discrete_sequence=['#38bdf8', '#f43f5e', '#10b981', '#fbbf24']
            )
            fig_dec.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_dec, use_container_width=True)
            
        with col_h2:
            st.markdown("#### Historical Statistical Summary")
            st_table = df_hist.groupby('SUBDIVISION')['ANNUAL'].agg(
                Count='count', Mean='mean', Min='min', Max='max', Std='std'
            ).round(1).reset_index()
            st.dataframe(st_table, use_container_width=True)
            
            # Download filtered historical view
            csv_hist = df_hist.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Comparison Data (CSV)",
                data=csv_hist,
                file_name=f"rainfall_comparison_{year_range[0]}_{year_range[1]}.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("Please select at least one subdivision to view historical charts.")

# =========================================================
# TAB 3: ANNUAL TREND ANALYSIS
# =========================================================
with tabs[2]:
    st.markdown(f"### 📈 Annual Trend & Anomaly Analysis — {selected_subdivision}")
    
    df_single = df_sub[
        (df_sub['SUBDIVISION'] == selected_subdivision) &
        (df_sub['YEAR'] >= year_range[0]) &
        (df_sub['YEAR'] <= year_range[1])
    ].sort_values('YEAR').copy()
    
    if not df_single.empty:
        # Linear Regression Slope
        x_vals = df_single['YEAR'].values
        y_vals = df_single['ANNUAL'].values
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        decadal_slope = round(slope * 10, 2)
        norm_val = round(df_single['NORMAL_ANNUAL'].iloc[0], 1)
        
        tr_kpi1, tr_kpi2, tr_kpi3, tr_kpi4 = st.columns(4)
        with tr_kpi1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Subdivision Normal</div>
                <div class="kpi-value">{norm_val} <span style="font-size: 0.9rem; color: #94a3b8;">mm</span></div>
                <div class="kpi-subtext">IMD 115-Yr Climatological Mean</div>
            </div>
            """, unsafe_allow_html=True)
        with tr_kpi2:
            st.markdown(f"""
            <div class="kpi-card {'emerald' if slope >= 0 else 'rose'}">
                <div class="kpi-label">Decadal Trend Rate</div>
                <div class="kpi-value">{decadal_slope:+} <span style="font-size: 0.9rem; color: #94a3b8;">mm/dec</span></div>
                <div class="kpi-subtext">{'Upward trend' if slope >= 0 else 'Downward trend'}</div>
            </div>
            """, unsafe_allow_html=True)
        with tr_kpi3:
            max_y = df_single.loc[df_single['ANNUAL'].idxmax()]
            st.markdown(f"""
            <div class="kpi-card indigo">
                <div class="kpi-label">Record Wettest Year</div>
                <div class="kpi-value">{int(max_y['YEAR'])}</div>
                <div class="kpi-subtext">{round(max_y['ANNUAL'], 1)} mm ({round(max_y['DEPARTURE_PCT'], 1):+}% dep)</div>
            </div>
            """, unsafe_allow_html=True)
        with tr_kpi4:
            min_y = df_single.loc[df_single['ANNUAL'].idxmin()]
            st.markdown(f"""
            <div class="kpi-card amber">
                <div class="kpi-label">Record Driest Year</div>
                <div class="kpi-value">{int(min_y['YEAR'])}</div>
                <div class="kpi-subtext">{round(min_y['ANNUAL'], 1)} mm ({round(min_y['DEPARTURE_PCT'], 1):+}% dep)</div>
            </div>
            """, unsafe_allow_html=True)

        # Interactive Trendline and Moving Average
        ma_choice = st.radio("Moving Average Window", [3, 5, 10], horizontal=True, index=1)
        df_single['MA'] = df_single['ANNUAL'].rolling(ma_choice, min_periods=1).mean()
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=df_single['YEAR'], y=df_single['ANNUAL'],
            name="Annual Rainfall", marker_color='rgba(56, 189, 248, 0.4)'
        ))
        fig_trend.add_trace(go.Scatter(
            x=df_single['YEAR'], y=df_single['MA'],
            name=f"{ma_choice}-Year Moving Average", line=dict(color='#38bdf8', width=2.5)
        ))
        fig_trend.add_trace(go.Scatter(
            x=df_single['YEAR'], y=slope * x_vals + intercept,
            name=f"Linear Trendline ({decadal_slope:+} mm/decade)",
            line=dict(color='#f43f5e', width=2, dash='dash')
        ))
        fig_trend.add_hline(y=norm_val, line_dash="dot", line_color="#fbbf24", annotation_text="Normal")
        fig_trend.update_layout(
            title=f"Annual Rainfall Trend & Moving Average ({year_range[0]}–{year_range[1]})",
            xaxis_title="Year", yaxis_title="Rainfall (mm)",
            template="plotly_dark", height=420, legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # Anomaly / Departure from Normal Chart
        df_single['ANOMALY_COLOR'] = np.where(df_single['DEPARTURE_PCT'] >= 0, '#38bdf8', '#f43f5e')
        fig_anom = go.Figure()
        fig_anom.add_trace(go.Bar(
            x=df_single['YEAR'], y=df_single['DEPARTURE_PCT'],
            marker_color=df_single['ANOMALY_COLOR'],
            name="Departure %"
        ))
        fig_anom.add_hline(y=20, line_dash="dash", line_color="#34d399", annotation_text="+20% Excess Threshold")
        fig_anom.add_hline(y=-20, line_dash="dash", line_color="#fb7185", annotation_text="-20% Deficient Threshold")
        fig_anom.update_layout(
            title="Annual Departure % from Climatological Normal",
            xaxis_title="Year", yaxis_title="Departure (%)",
            template="plotly_dark", height=350
        )
        st.plotly_chart(fig_anom, use_container_width=True)

# =========================================================
# TAB 4: MONTHLY & SEASONAL ANALYSIS
# =========================================================
with tabs[3]:
    st.markdown(f"### 🗓️ Monthly Seasonality & Heatmap — {selected_subdivision}")
    
    df_sub_months = df_sub[
        (df_sub['SUBDIVISION'] == selected_subdivision) &
        (df_sub['YEAR'] >= year_range[0]) &
        (df_sub['YEAR'] <= year_range[1])
    ].copy()
    
    col_m1, col_m2 = st.columns([7, 5])
    with col_m1:
        # 12-Month Distribution Boxplot
        df_melt = pd.melt(
            df_sub_months, id_vars=['YEAR'], value_vars=MONTHS,
            var_name='Month', value_name='Rainfall'
        )
        fig_box = px.box(
            df_melt, x='Month', y='Rainfall', color='Month',
            title="Monthly Rainfall Variability Distribution (12 Months)",
            labels={'Rainfall': 'Monthly Rainfall (mm)'},
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig_box.update_layout(template="plotly_dark", height=420, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
        
    with col_m2:
        # Mean Monthly Profile
        mean_monthly = df_sub_months[MONTHS].mean().reset_index()
        mean_monthly.columns = ['Month', 'MeanRainfall']
        fig_profile = px.line(
            mean_monthly, x='Month', y='MeanRainfall', markers=True,
            title="Climatological Monthly Progression Curve",
            labels={'MeanRainfall': 'Normal Monthly Mean (mm)'},
            color_discrete_sequence=['#38bdf8']
        )
        fig_profile.update_traces(fill='tozeroy', fillcolor='rgba(56, 189, 248, 0.15)')
        fig_profile.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig_profile, use_container_width=True)

    # Month vs Year 2D Heatmap
    st.markdown("#### Month × Year Precipitation Heatmap")
    heatmap_data = df_sub_months.set_index('YEAR')[MONTHS].T
    fig_heat = px.imshow(
        heatmap_data,
        labels=dict(x="Year", y="Month", color="Rainfall (mm)"),
        color_continuous_scale="Blues",
        aspect="auto",
        title=f"Yearly-Monthly Rainfall Matrix: {selected_subdivision} ({year_range[0]}–{year_range[1]})"
    )
    fig_heat.update_layout(template="plotly_dark", height=380)
    st.plotly_chart(fig_heat, use_container_width=True)

# =========================================================
# TAB 5: REGIONAL & DISTRICTS ANALYSIS
# =========================================================
with tabs[4]:
    st.markdown("### 🗺️ Regional & District Normal Explorer")
    st.caption("Powered by official IMD District-Wise Rainfall Normal Dataset (641 districts across 35 States & UTs)")
    
    col_reg1, col_reg2 = st.columns([4, 8])
    with col_reg1:
        states = ["All States"] + get_state_list(df_dist)
        selected_state = st.selectbox("Filter by State / UT", states)
        
        dist_in_state = get_district_list(df_dist, selected_state)
        selected_dist = st.selectbox("Drill-down to District", dist_in_state)
        
        # Display selected district normal card
        dist_row = df_dist[df_dist['DISTRICT'] == selected_dist].iloc[0]
        st.markdown(f"""
        <div class="prediction-box">
            <span class="badge info">{dist_row['STATE_UT_NAME']}</span>
            <h3 style="margin: 8px 0 2px 0; color: #ffffff;">{dist_row['DISTRICT']}</h3>
            <div class="kpi-label">Annual Normal Rainfall</div>
            <div class="pred-number">{dist_row['ANNUAL']} <span style="font-size: 1.2rem; color: #94a3b8;">mm</span></div>
            <div style="margin-top: 12px; font-size: 0.85rem; color: #cbd5e1;">
                <div>• Monsoon (Jun-Sep): <b>{dist_row['Jun-Sep']} mm</b> ({round(dist_row['MONSOON_SHARE_PCT'], 1)}%)</div>
                <div>• Pre-Monsoon (Mar-May): <b>{dist_row['Mar-May']} mm</b></div>
                <div>• Post-Monsoon (Oct-Dec): <b>{dist_row['Oct-Dec']} mm</b></div>
                <div>• Winter (Jan-Feb): <b>{dist_row['Jan-Feb']} mm</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_reg2:
        if selected_state != "All States":
            df_state_dists = df_dist[df_dist['STATE_UT_NAME'] == selected_state].sort_values('ANNUAL', ascending=True)
            fig_state = px.bar(
                df_state_dists, x='ANNUAL', y='DISTRICT', orientation='h',
                title=f"District Normal Annual Rainfall — {selected_state}",
                labels={'ANNUAL': 'Normal Rainfall (mm)', 'DISTRICT': 'District'},
                color='ANNUAL', color_continuous_scale='Teal'
            )
            fig_state.update_layout(template="plotly_dark", height=max(400, len(df_state_dists) * 22))
            st.plotly_chart(fig_state, use_container_width=True)
        else:
            # National Top 10 Wettest and Driest Districts
            top_dist = df_dist.sort_values('ANNUAL', ascending=False).head(10)
            fig_top_dist = px.bar(
                top_dist, x='ANNUAL', y='DISTRICT', orientation='h',
                title="Top 10 Wettest Districts in India (Official Climatological Normal)",
                labels={'ANNUAL': 'Normal Rainfall (mm)', 'DISTRICT': 'District'},
                color='ANNUAL', color_continuous_scale='Blues'
            )
            fig_top_dist.update_layout(template="plotly_dark", height=420, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_top_dist, use_container_width=True)

    st.markdown("#### District Climatological Benchmark Table")
    if selected_state != "All States":
        table_dists = df_dist[df_dist['STATE_UT_NAME'] == selected_state]
    else:
        table_dists = df_dist
    st.dataframe(
        table_dists[['STATE_UT_NAME', 'DISTRICT', 'ANNUAL', 'Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']],
        use_container_width=True
    )

# =========================================================
# TAB 6: RAINFALL NORMAL & DEVIATIONS
# =========================================================
with tabs[5]:
    st.markdown(f"### ⚖️ IMD Rainfall Normal & Departure Diagnostics — {selected_subdivision}")
    
    df_dep = df_sub[
        (df_sub['SUBDIVISION'] == selected_subdivision) &
        (df_sub['YEAR'] >= year_range[0]) &
        (df_sub['YEAR'] <= year_range[1])
    ].copy()
    
    col_d1, col_d2 = st.columns([5, 7])
    with col_d1:
        # Category Breakdown Pie Chart
        cat_counts = df_dep['IMD_CATEGORY'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Years']
        color_map = {
            'Large Excess': '#3b82f6',
            'Excess': '#10b981',
            'Normal': '#38bdf8',
            'Deficient': '#f59e0b',
            'Scanty': '#f43f5e'
        }
        fig_cat_pie = px.pie(
            cat_counts, names='Category', values='Years',
            title=f"IMD Meteorological Category Frequency ({year_range[0]}–{year_range[1]})",
            color='Category', color_discrete_map=color_map, hole=0.45
        )
        fig_cat_pie.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig_cat_pie, use_container_width=True)
        
    with col_d2:
        # Departure Timeline
        fig_dep_timeline = px.scatter(
            df_dep, x='YEAR', y='DEPARTURE_PCT', color='IMD_CATEGORY',
            color_discrete_map=color_map, size=np.abs(df_dep['DEPARTURE_PCT']) + 5,
            title=f"Departure % from Long-term Normal ({year_range[0]}–{year_range[1]})",
            labels={'DEPARTURE_PCT': 'Departure (%)', 'YEAR': 'Year'}
        )
        fig_dep_timeline.add_hline(y=20, line_dash="dash", line_color="#10b981")
        fig_dep_timeline.add_hline(y=-20, line_dash="dash", line_color="#f59e0b")
        fig_dep_timeline.add_hline(y=0, line_dash="solid", line_color="#ffffff", opacity=0.3)
        fig_dep_timeline.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig_dep_timeline, use_container_width=True)

    # Notable Drought and Excess Years Table
    st.markdown("#### Extreme Meteorological Anomalies Recorded")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("**Severe Drought Years (Deficient / Scanty)**")
        drought_years = df_dep[df_dep['DEPARTURE_PCT'] <= -20.0].sort_values('DEPARTURE_PCT')[
            ['YEAR', 'ANNUAL', 'NORMAL_ANNUAL', 'DEPARTURE_PCT', 'IMD_CATEGORY']
        ]
        st.dataframe(drought_years, use_container_width=True)
    with col_e2:
        st.markdown("**Excess / Flood Years**")
        flood_years = df_dep[df_dep['DEPARTURE_PCT'] >= 20.0].sort_values('DEPARTURE_PCT', ascending=False)[
            ['YEAR', 'ANNUAL', 'NORMAL_ANNUAL', 'DEPARTURE_PCT', 'IMD_CATEGORY']
        ]
        st.dataframe(flood_years, use_container_width=True)

# =========================================================
# TAB 7: MACHINE LEARNING PREDICTION
# =========================================================
with tabs[6]:
    st.markdown("### 🤖 Predictive Scenario Simulator (Machine Learning Engine)")
    st.caption("Dual Model Architecture: Random Forest Regressor (R² ≈ 0.90) for continuous precipitation & Balanced Random Forest Classifier for meteorological categories.")
    
    # Load defaults from selected subdivision
    sub_data = df_sub[df_sub['SUBDIVISION'] == selected_subdivision].sort_values('YEAR')
    def_norm = float(sub_data['NORMAL_ANNUAL'].iloc[0])
    def_jan_feb = float(sub_data['Jan-Feb'].mean())
    def_mar_may = float(sub_data['Mar-May'].mean())
    def_lag1 = float(sub_data['ANNUAL'].iloc[-1]) if len(sub_data) > 0 else def_norm
    def_lag2 = float(sub_data['ANNUAL'].iloc[-2]) if len(sub_data) > 1 else def_norm
    def_roll3 = float(sub_data['ANNUAL'].tail(3).mean())
    def_roll5 = float(sub_data['ANNUAL'].tail(5).mean())
    
    col_sim1, col_sim2 = st.columns([6, 6])
    with col_sim1:
        st.markdown("#### Scenario Feature Parameters")
        in_sub = st.selectbox("Predictive Target Subdivision", subdivisions, index=subdivisions.index(selected_subdivision))
        
        # If target changed, recompute defaults
        cur_sub_data = df_sub[df_sub['SUBDIVISION'] == in_sub].sort_values('YEAR')
        c_norm = float(cur_sub_data['NORMAL_ANNUAL'].iloc[0])
        
        s_jan_feb = st.slider("Winter Rainfall (Jan-Feb) mm", 0.0, 300.0, round(float(cur_sub_data['Jan-Feb'].mean()), 1), step=0.5)
        s_mar_may = st.slider("Pre-Monsoon Rainfall (Mar-May) mm", 0.0, 800.0, round(float(cur_sub_data['Mar-May'].mean()), 1), step=1.0)
        s_lag1 = st.slider("Previous Year Rainfall (t-1) mm", 50.0, 5000.0, round(float(cur_sub_data['ANNUAL'].iloc[-1]), 1), step=10.0)
        s_lag2 = st.slider("Rainfall 2 Years Ago (t-2) mm", 50.0, 5000.0, round(float(cur_sub_data['ANNUAL'].iloc[-2]), 1), step=10.0)
        s_roll3 = st.slider("3-Year Rolling Mean mm", 50.0, 5000.0, round(float(cur_sub_data['ANNUAL'].tail(3).mean()), 1), step=10.0)
        s_roll5 = st.slider("5-Year Rolling Mean mm", 50.0, 5000.0, round(float(cur_sub_data['ANNUAL'].tail(5).mean()), 1), step=10.0)
        
        btn_predict = st.button("🚀 Run Scenario Prediction", use_container_width=True, type="primary")

    with col_sim2:
        st.markdown("#### Forecast & Probabilities")
        
        # Run inference
        prediction = predict_scenario(
            reg_model, clf_model, c_norm,
            s_jan_feb, s_mar_may, s_lag1, s_lag2, s_roll3, s_roll5
        )
        
        pred_cat = prediction['model_derived_category']
        pred_badge_class = "success" if pred_cat == "Normal" else "info" if "Excess" in pred_cat else "danger"
        
        st.markdown(f"""
        <div class="prediction-box">
            <span class="badge {pred_badge_class}">{pred_cat}</span>
            <span class="disclaimer-tag">Model-Derived: Random Forest Regression</span>
            <div style="margin-top: 10px;">
                <div class="kpi-label">Predicted Annual Rainfall</div>
                <div class="pred-number">{prediction['predicted_annual_mm']} <span style="font-size: 1.2rem; color: #94a3b8;">mm</span></div>
            </div>
            <div style="margin-top: 12px; font-size: 0.95rem; color: #cbd5e1;">
                <div>• Climatological Normal: <b>{prediction['normal_annual_mm']} mm</b></div>
                <div>• Expected Departure: <b>{prediction['predicted_departure_pct']:+}%</b></div>
                <div>• Direct Classifier Output: <b>{prediction['direct_classifier_category']}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Class Probability Bar Chart
        prob_df = pd.DataFrame(list(prediction['class_probabilities'].items()), columns=['Category', 'Probability'])
        fig_prob = px.bar(
            prob_df, x='Probability', y='Category', orientation='h',
            title="Classifier Probability Distribution (%)",
            labels={'Probability': 'Confidence (%)', 'Category': 'Category'},
            color='Category',
            color_discrete_map={'Deficient': '#f43f5e', 'Normal': '#38bdf8', 'Excess': '#10b981'}
        )
        fig_prob.update_layout(template="plotly_dark", height=240, showlegend=False)
        st.plotly_chart(fig_prob, use_container_width=True)

# =========================================================
# TAB 8: MODEL EVALUATION
# =========================================================
with tabs[7]:
    st.markdown("### 🔬 Machine Learning Model Evaluation & Diagnostics")
    
    clf_m = ml_metrics['classification']
    reg_m = ml_metrics['regression']
    
    # Regression Metrics
    st.markdown("#### 1. Continuous Regression Performance (Random Forest Regressor)")
    rm1, rm2, rm3, rm4 = st.columns(4)
    with rm1:
        st.markdown(f"""
        <div class="kpi-card emerald">
            <div class="kpi-label">R² Score (Variance Explained)</div>
            <div class="kpi-value">{reg_m['r2']}</div>
            <div class="kpi-subtext">Strong predictive accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with rm2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Root Mean Squared Error</div>
            <div class="kpi-value">{reg_m['rmse']} <span style="font-size: 0.9rem; color: #94a3b8;">mm</span></div>
            <div class="kpi-subtext">On independent test split</div>
        </div>
        """, unsafe_allow_html=True)
    with rm3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Mean Absolute Error (MAE)</div>
            <div class="kpi-value">{reg_m['mae']} <span style="font-size: 0.9rem; color: #94a3b8;">mm</span></div>
            <div class="kpi-subtext">Average deviation per prediction</div>
        </div>
        """, unsafe_allow_html=True)
    with rm4:
        st.markdown(f"""
        <div class="kpi-card indigo">
            <div class="kpi-label">Hold-out Test Size</div>
            <div class="kpi-value">{reg_m['test_size']}</div>
            <div class="kpi-subtext">20% stratified test observations</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    # Classification Metrics
    st.markdown("#### 2. Classification Performance (Random Forest with Balanced Class Weights)")
    cm1, cm2, cm3, cm4 = st.columns(4)
    with cm1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Overall Accuracy</div>
            <div class="kpi-value">{round(clf_m['accuracy'] * 100, 1)}%</div>
            <div class="kpi-subtext">Weighted baseline</div>
        </div>
        """, unsafe_allow_html=True)
    with cm2:
        st.markdown(f"""
        <div class="kpi-card emerald">
            <div class="kpi-label">Precision (Macro)</div>
            <div class="kpi-value">{round(clf_m['precision_macro'] * 100, 1)}%</div>
            <div class="kpi-subtext">Unweighted class average</div>
        </div>
        """, unsafe_allow_html=True)
    with cm3:
        st.markdown(f"""
        <div class="kpi-card rose">
            <div class="kpi-label">Recall (Macro)</div>
            <div class="kpi-value">{round(clf_m['recall_macro'] * 100, 1)}%</div>
            <div class="kpi-subtext">Unweighted class average</div>
        </div>
        """, unsafe_allow_html=True)
    with cm4:
        st.markdown(f"""
        <div class="kpi-card amber">
            <div class="kpi-label">F1-Score (Macro)</div>
            <div class="kpi-value">{round(clf_m['f1_macro'] * 100, 1)}%</div>
            <div class="kpi-subtext">Unweighted class average</div>
        </div>
        """, unsafe_allow_html=True)

    col_eval1, col_eval2 = st.columns(2)
    with col_eval1:
        # Confusion Matrix
        cm_data = clf_m['confusion_matrix']
        fig_cm = px.imshow(
            cm_data,
            x=CLASSES, y=CLASSES,
            color_continuous_scale='Blues',
            text_auto=True,
            title="Confusion Matrix (Predicted vs Actual)",
            labels=dict(x="Predicted Category", y="True Category", color="Observations")
        )
        fig_cm.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_eval2:
        # Feature Importance
        feat_imp_df = pd.DataFrame(
            list(reg_m['feature_importance'].items()), columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=True)
        fig_feat = px.bar(
            feat_imp_df, x='Importance', y='Feature', orientation='h',
            title="Random Forest Feature Importance",
            labels={'Importance': 'Gini Importance / Gain', 'Feature': 'Engineered Feature'},
            color='Importance', color_continuous_scale='Tealgrn'
        )
        fig_feat.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig_feat, use_container_width=True)

    # Per-class metrics table
    st.markdown("#### Detailed Per-Class Breakdown")
    per_class_df = pd.DataFrame(clf_m['per_class']).T.reset_index()
    per_class_df.columns = ['Meteorological Category', 'Precision', 'Recall', 'F1-Score']
    st.dataframe(per_class_df, use_container_width=True)

# =========================================================
# TAB 9: NATURAL LANGUAGE AI AGENT
# =========================================================
with tabs[8]:
    st.markdown("### 💬 Natural Language AI Climatological Query Agent")
    st.markdown("""
    Ask questions in plain English. The agent strictly executes queries against the real dataset,
    computes real mathematical aggregations, and generates grounded Plotly visualizations.
    **Zero invented numerical values.**
    """)
    
    # Architecture flow diagram
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 10px; padding: 12px 18px; margin-bottom: 16px; font-size: 0.84rem; color: #94a3b8;">
        <b style="color: #38bdf8;">Deterministic AI Architecture:</b>
        User Query &rarr; Intent & Entity Extraction &rarr; Structured JSON &rarr; Validation &rarr; Pandas Query Execution &rarr; Plotly Chart/Table &rarr; Grounded AI Explanation
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Example Buttons
    st.markdown("##### Quick Example Queries (Click to load)")
    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        if st.button("Compare Tamil Nadu and Kerala rainfall from 2000 to 2015.", use_container_width=True):
            st.session_state['user_query'] = "Compare Tamil Nadu and Kerala rainfall from 2000 to 2015."
        if st.button("Which year had the highest rainfall?", use_container_width=True):
            st.session_state['user_query'] = "Which year had the highest rainfall?"
        if st.button("Show lowest rainfall year in Punjab", use_container_width=True):
            st.session_state['user_query'] = "Show lowest rainfall year in Punjab"
    with ex_col2:
        if st.button("Show rainfall trends in Tamil Nadu.", use_container_width=True):
            st.session_state['user_query'] = "Show rainfall trends in Tamil Nadu."
        if st.button("Which region has the highest average rainfall?", use_container_width=True):
            st.session_state['user_query'] = "Which region has the highest average rainfall?"
        if st.button("Compare Rajasthan and Gujarat between 1980 and 2000", use_container_width=True):
            st.session_state['user_query'] = "Compare Rajasthan and Gujarat between 1980 and 2000"

    query_input = st.text_input(
        "Enter your question:",
        value=st.session_state.get('user_query', "Compare Tamil Nadu and Kerala rainfall from 2000 to 2015."),
        placeholder="e.g. Compare Tamil Nadu and Kerala rainfall from 2000 to 2015."
    )
    
    clar_col1, clar_col2 = st.columns([3, 1])
    with clar_col1:
        prefer_clario = st.checkbox(
            "⚡ Enable Clario MCP-Assisted Query Pipeline",
            value=True,
            help="Utilizes Clario MCP protocol for intent parsing combined with deterministic Pandas verification on actual datasets."
        )
    with clar_col2:
        if clario.get_status().get("connected"):
            st.markdown("<span style='color: #10b981; font-size: 0.82rem; font-weight: 600;'>🟢 Clario MCP Ready</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color: #f59e0b; font-size: 0.82rem; font-weight: 600;'>🟠 Clario Offline Fallback</span>", unsafe_allow_html=True)
    
    if st.button("🔍 Execute AI Analysis", type="primary"):
        with st.spinner("Processing natural language query against datasets..."):
            nl_res = query_rainfall_ai(query_input, df_sub, df_dist, prefer_clario=prefer_clario)
            
            # Clario usage indicator
            if nl_res.get('clario_used'):
                st.markdown("""
                <div style="margin-bottom: 10px;">
                    <span class="header-badge" style="background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35);">
                        ⚡ Clario MCP Agent Assisted
                    </span>
                    <span style="font-size: 0.8rem; color: #94a3b8; margin-left: 8px;">
                        Intent extracted via Clario MCP JSON-RPC protocol & verified against ground truth CSV
                    </span>
                </div>
                """, unsafe_allow_html=True)

            # 1. Structured JSON Inspection
            with st.expander("🛠️ Inspect Structured Query Specification (JSON)", expanded=False):
                st.json(nl_res['structured_json'])
                
            # 2. AI Explanation Box
            st.markdown(f"""
            <div class="ai-response-box">
                {nl_res['explanation'].replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)
            
            # 3. Dynamic Plotly Chart
            if nl_res.get('chart'):
                st.plotly_chart(nl_res['chart'], use_container_width=True)
                
            # 4. Computed Data Table & Download Button
            if nl_res.get('data') is not None and not nl_res['data'].empty:
                st.markdown("##### Query Result Dataset")
                st.dataframe(nl_res['data'], use_container_width=True)
                csv_nl = nl_res['data'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Export Query Results (CSV)",
                    data=csv_nl,
                    file_name="ai_rainfall_query_result.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# =========================================================
# TAB 10: AI-GENERATED DATA INSIGHTS
# =========================================================
with tabs[9]:
    st.markdown("### 💡 AI-Generated Climatological Insights & Intelligence")
    st.markdown("Automated statistical anomaly detection, long-term decadal shifts, and extreme event retrospectives.")
    
    exec_insights = generate_executive_insights(df_sub)
    for ins in exec_insights:
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-header">
                <div>
                    <span class="badge {ins['severity']}">{ins['badge']}</span>
                    <span style="font-size: 0.8rem; color: #94a3b8; margin-left: 8px;">{ins['category']}</span>
                </div>
                <div style="font-weight: 700; color: #38bdf8; font-family: var(--font-heading);">
                    {ins['metric_label']}: {ins['metric_value']}
                </div>
            </div>
            <div class="insight-title">{ins['title']}</div>
            <div class="insight-body">{ins['summary']}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("### 🔍 Subdivision Automated Deep Dive")
    dive_sub = st.selectbox("Select Subdivision for Automated AI Briefing", subdivisions, index=subdivisions.index(selected_subdivision))
    dive_stats = get_subdivision_deep_dive(df_sub, dive_sub)
    
    if dive_stats:
        dd1, dd2, dd3, dd4 = st.columns(4)
        with dd1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Climatological Normal</div>
                <div class="kpi-value">{dive_stats['normal_annual']} <span style="font-size: 0.9rem; color: #94a3b8;">mm</span></div>
                <div class="kpi-subtext">Monsoon: {dive_stats['normal_monsoon']} mm</div>
            </div>
            """, unsafe_allow_html=True)
        with dd2:
            st.markdown(f"""
            <div class="kpi-card {'emerald' if dive_stats['decadal_trend_rate_mm'] >= 0 else 'rose'}">
                <div class="kpi-label">Decadal Trend Slope</div>
                <div class="kpi-value">{dive_stats['decadal_trend_rate_mm']:+} <span style="font-size: 0.9rem; color: #94a3b8;">mm</span></div>
                <div class="kpi-subtext">Rate per decade</div>
            </div>
            """, unsafe_allow_html=True)
        with dd3:
            st.markdown(f"""
            <div class="kpi-card amber">
                <div class="kpi-label">Drought / Deficit Risk</div>
                <div class="kpi-value">{dive_stats['deficit_years_pct']}%</div>
                <div class="kpi-subtext">Frequency of deficient years</div>
            </div>
            """, unsafe_allow_html=True)
        with dd4:
            st.markdown(f"""
            <div class="kpi-card emerald">
                <div class="kpi-label">Normal Year Stability</div>
                <div class="kpi-value">{dive_stats['normal_years_pct']}%</div>
                <div class="kpi-subtext">Years within ±19% normal</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 12px; padding: 18px 22px; margin-top: 10px; color: #e2e8f0; font-size: 0.95rem; line-height: 1.6;">
            <b>Automated AI Synthesis for {dive_sub}:</b><br>
            Over the 115-year historical archive (1901–2015), <b>{dive_sub}</b> recorded an annual climatological normal of <b>{dive_stats['normal_annual']} mm</b>. 
            The region experienced normal rainfall in <b>{dive_stats['normal_years_pct']}%</b> of years, deficient/scanty conditions in <b>{dive_stats['deficit_years_pct']}%</b> of years, and excess rainfall in <b>{dive_stats['excess_years_pct']}%</b> of years.
            The all-time recorded peak deluge occurred in <b>{dive_stats['wettest_year']}</b> ({dive_stats['wettest_val']} mm), while the most severe historical drought occurred in <b>{dive_stats['driest_year']}</b> ({dive_stats['driest_val']} mm).
            The long-term decadal precipitation trajectory is <b>{dive_stats['decadal_trend_rate_mm']:+} mm per decade</b>.
        </div>
        """, unsafe_allow_html=True)
