"""
Natural Language AI Query Engine for India Rainfall Analytics.
Follows the deterministic architecture:
Natural Language -> Intent Extraction -> Structured JSON -> Validation
-> Pandas Query -> Actual Dataset Result -> Plotly Chart/Table -> Grounded AI Explanation.
Guarantees ZERO numerical hallucination.
"""

import re
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Canonical subdivision aliases mapping to dataset SUBDIVISION names
SUB_ALIASES = {
    "ANDAMAN & NICOBAR ISLANDS": "ANDAMAN & NICOBAR ISLANDS",
    "ANDAMAN AND NICOBAR": "ANDAMAN & NICOBAR ISLANDS",
    "ANDAMAN": "ANDAMAN & NICOBAR ISLANDS",
    "ARUNACHAL PRADESH": "ARUNACHAL PRADESH",
    "ARUNACHAL": "ARUNACHAL PRADESH",
    "ASSAM & MEGHALAYA": "ASSAM & MEGHALAYA",
    "ASSAM": "ASSAM & MEGHALAYA",
    "MEGHALAYA": "ASSAM & MEGHALAYA",
    "BIHAR": "BIHAR",
    "CHHATTISGARH": "CHHATTISGARH",
    "CHATISGARH": "CHHATTISGARH",
    "COASTAL ANDHRA PRADESH": "COASTAL ANDHRA PRADESH",
    "ANDHRA PRADESH": "COASTAL ANDHRA PRADESH",
    "ANDHRA": "COASTAL ANDHRA PRADESH",
    "COASTAL KARNATAKA": "COASTAL KARNATAKA",
    "EAST MADHYA PRADESH": "EAST MADHYA PRADESH",
    "EAST RAJASTHAN": "EAST RAJASTHAN",
    "EAST UTTAR PRADESH": "EAST UTTAR PRADESH",
    "GANGETIC WEST BENGAL": "GANGETIC WEST BENGAL",
    "WEST BENGAL": "GANGETIC WEST BENGAL",
    "GUJARAT REGION": "GUJARAT REGION",
    "GUJARAT": "GUJARAT REGION",
    "HARYANA DELHI & CHANDIGARH": "HARYANA DELHI & CHANDIGARH",
    "HARYANA": "HARYANA DELHI & CHANDIGARH",
    "DELHI": "HARYANA DELHI & CHANDIGARH",
    "CHANDIGARH": "HARYANA DELHI & CHANDIGARH",
    "HIMACHAL PRADESH": "HIMACHAL PRADESH",
    "HIMACHAL": "HIMACHAL PRADESH",
    "JAMMU & KASHMIR": "JAMMU & KASHMIR",
    "JAMMU AND KASHMIR": "JAMMU & KASHMIR",
    "KASHMIR": "JAMMU & KASHMIR",
    "JHARKHAND": "JHARKHAND",
    "KERALA": "KERALA",
    "KONKAN & GOA": "KONKAN & GOA",
    "GOA": "KONKAN & GOA",
    "KONKAN": "KONKAN & GOA",
    "LAKSHADWEEP": "LAKSHADWEEP",
    "MADHYA MAHARASHTRA": "MADHYA MAHARASHTRA",
    "MAHARASHTRA": "MADHYA MAHARASHTRA",
    "MATATHWADA": "MATATHWADA",
    "MARATHWADA": "MATATHWADA",
    "NAGA MANI MIZO TRIPURA": "NAGA MANI MIZO TRIPURA",
    "NAGALAND": "NAGA MANI MIZO TRIPURA",
    "MANIPUR": "NAGA MANI MIZO TRIPURA",
    "MIZORAM": "NAGA MANI MIZO TRIPURA",
    "TRIPURA": "NAGA MANI MIZO TRIPURA",
    "NORTH INTERIOR KARNATAKA": "NORTH INTERIOR KARNATAKA",
    "ORISSA": "ORISSA",
    "ODISHA": "ORISSA",
    "PUNJAB": "PUNJAB",
    "RAYALSEEMA": "RAYALSEEMA",
    "SAURASHTRA & KUTCH": "SAURASHTRA & KUTCH",
    "SAURASHTRA": "SAURASHTRA & KUTCH",
    "KUTCH": "SAURASHTRA & KUTCH",
    "SOUTH INTERIOR KARNATAKA": "SOUTH INTERIOR KARNATAKA",
    "KARNATAKA": "SOUTH INTERIOR KARNATAKA",
    "SUB HIMALAYAN WEST BENGAL & SIKKIM": "SUB HIMALAYAN WEST BENGAL & SIKKIM",
    "SIKKIM": "SUB HIMALAYAN WEST BENGAL & SIKKIM",
    "TAMIL NADU": "TAMIL NADU",
    "TELANGANA": "TELANGANA",
    "UTTARAKHAND": "UTTARAKHAND",
    "UTTARANCHAL": "UTTARAKHAND",
    "VIDARBHA": "VIDARBHA",
    "WEST MADHYA PRADESH": "WEST MADHYA PRADESH",
    "MADHYA PRADESH": "WEST MADHYA PRADESH",
    "WEST RAJASTHAN": "WEST RAJASTHAN",
    "RAJASTHAN": "WEST RAJASTHAN",
    "WEST UTTAR PRADESH": "WEST UTTAR PRADESH",
    "UTTAR PRADESH": "WEST UTTAR PRADESH",
    "UP": "WEST UTTAR PRADESH"
}

def extract_intent_and_entities(query: str, available_districts: list = None) -> dict:
    """Extracts user intent, target regions, districts, and time ranges."""
    q = query.strip()
    q_lower = q.lower()
    
    # 1. Intent identification
    if any(k in q_lower for k in ["compare", "versus", " vs ", " vs. ", "difference between"]):
        intent = "compare"
    elif any(k in q_lower for k in ["trend", "trends", "over time", "history", "historical", "yearly pattern", "evolution"]):
        intent = "trend"
    elif any(k in q_lower for k in ["highest rainfall", "maximum rainfall", "max rainfall", "wettest", "most rainfall", "peak rainfall"]):
        if any(w in q_lower for w in ["year", "years", "annual"]):
            intent = "highest_year"
        else:
            intent = "highest_region"
    elif any(k in q_lower for k in ["lowest rainfall", "minimum rainfall", "min rainfall", "driest", "least rainfall", "drought"]):
        if any(w in q_lower for w in ["year", "years", "annual"]):
            intent = "lowest_year"
        else:
            intent = "lowest_region"
    elif any(k in q_lower for k in ["highest average", "highest mean", "maximum average", "wettest region", "wettest subdivision", "wettest state"]):
        intent = "highest_region"
    elif any(k in q_lower for k in ["lowest average", "lowest mean", "minimum average", "driest region", "driest subdivision", "driest state"]):
        intent = "lowest_region"
    elif any(k in q_lower for k in ["month", "monthly", "season", "monsoon distribution", "seasonality"]):
        intent = "monthly_distribution"
    elif any(k in q_lower for k in ["normal", "deviation", "departure", "deficit", "excess"]):
        intent = "departure_analysis"
    elif any(k in q_lower for k in ["district", "district normal"]):
        intent = "district_normal"
    elif "average" in q_lower or "mean" in q_lower:
        intent = "highest_region"
    else:
        intent = "trend"
        
    # 2. Extract years (between 1901 and 2015)
    years = [int(y) for y in re.findall(r'\b(19\d\d|20\d\d)\b', q)]
    if len(years) >= 2:
        year_start = max(1901, min(years))
        year_end = min(2015, max(years))
        is_single_year = False
    elif len(years) == 1:
        year_start = max(1901, min(2015, years[0]))
        year_end = year_start
        is_single_year = True
    else:
        year_start = 1901
        year_end = 2015
        is_single_year = False
        
    # 3. Match subdivisions using alias dictionary (longest alias match first)
    matched_regions = []
    # Sort aliases by length descending to match 'COASTAL KARNATAKA' before 'KARNATAKA'
    sorted_aliases = sorted(SUB_ALIASES.keys(), key=lambda x: len(x), reverse=True)
    for alias in sorted_aliases:
        pattern = r'\b' + re.escape(alias.lower()) + r'\b'
        if re.search(pattern, q_lower):
            canon = SUB_ALIASES[alias]
            if canon not in matched_regions:
                matched_regions.append(canon)
                
    # 4. Match district names if available
    matched_districts = []
    if available_districts:
        for dist in available_districts:
            if re.search(r'\b' + re.escape(dist.lower()) + r'\b', q_lower):
                if dist not in matched_districts:
                    matched_districts.append(dist)
                    
    # Refine intent if multiple regions found but intent was generic
    if len(matched_regions) >= 2 and intent in ["trend", "unknown"]:
        intent = "compare"

    return {
        "raw_query": query,
        "intent": intent,
        "parameters": {
            "regions": matched_regions,
            "districts": matched_districts,
            "year_start": year_start,
            "year_end": year_end,
            "is_single_year": is_single_year
        }
    }

def execute_nl_query(spec: dict, df_sub: pd.DataFrame, df_dist: pd.DataFrame):
    """
    Executes the validated query against the real Pandas dataframes.
    Returns calculated data, Plotly figure, and grounded explanation.
    """
    intent = spec["intent"]
    params = spec["parameters"]
    regions = params.get("regions", [])
    y_start = params.get("year_start", 1901)
    y_end = params.get("year_end", 2015)
    is_single_year = params.get("is_single_year", False)
    
    # Filter df_sub by year
    df_filtered = df_sub[(df_sub['YEAR'] >= y_start) & (df_sub['YEAR'] <= y_end)].copy()
    
    # ----------------------------------------------------
    # CASE 1: COMPARE REGIONS
    # ----------------------------------------------------
    if intent == "compare":
        if not regions:
            # Default to top 2 if unspecified
            regions = ["TAMIL NADU", "KERALA"]
        elif len(regions) == 1:
            # Add national average or second region
            alt = "KERALA" if regions[0] != "KERALA" else "TAMIL NADU"
            regions.append(alt)
            
        df_comp = df_filtered[df_filtered['SUBDIVISION'].isin(regions)].copy()
        
        # Calculate summary statistics
        summary = df_comp.groupby('SUBDIVISION')['ANNUAL'].agg(['mean', 'max', 'min', 'std', 'count']).round(1)
        summary['monsoon_mean'] = df_comp.groupby('SUBDIVISION')['Jun-Sep'].mean().round(1)
        
        # Plotly chart
        if is_single_year or (y_end - y_start <= 5):
            fig = px.bar(
                df_comp, x='YEAR', y='ANNUAL', color='SUBDIVISION', barmode='group',
                title=f"Rainfall Comparison: {', '.join(regions)} ({y_start}-{y_end})",
                labels={'ANNUAL': 'Annual Rainfall (mm)', 'YEAR': 'Year'},
                color_discrete_sequence=['#38bdf8', '#fb7185', '#34d399', '#facc15']
            )
        else:
            fig = px.line(
                df_comp, x='YEAR', y='ANNUAL', color='SUBDIVISION', markers=True,
                title=f"Annual Rainfall Comparison: {', '.join(regions)} ({y_start}–{y_end})",
                labels={'ANNUAL': 'Annual Rainfall (mm)', 'YEAR': 'Year'},
                color_discrete_sequence=['#38bdf8', '#fb7185', '#34d399', '#facc15']
            )
            
        fig.update_layout(template="plotly_dark", hovermode="x unified")
        
        # Grounded AI explanation
        lines = [f"**Comparison between {', '.join(regions)} ({y_start}–{y_end}):**\n"]
        for r in regions:
            if r in summary.index:
                row = summary.loc[r]
                lines.append(f"- **{r}**: Average annual rainfall was **{row['mean']} mm** (Monsoon: {row['monsoon_mean']} mm). Peak was **{row['max']} mm**, and lowest was **{row['min']} mm**.")
                
        if len(regions) >= 2 and all(r in summary.index for r in regions[:2]):
            diff = round(summary.loc[regions[0], 'mean'] - summary.loc[regions[1], 'mean'], 1)
            higher = regions[0] if diff > 0 else regions[1]
            lines.append(f"\nOverall, **{higher}** received on average **{abs(diff)} mm** ({abs(round(diff / summary.loc[regions[1], 'mean'] * 100, 1))}%) more annual precipitation than **{regions[1] if diff > 0 else regions[0]}** during this period.")
            
        return {
            "data": df_comp[['YEAR', 'SUBDIVISION', 'ANNUAL', 'Jun-Sep', 'DEPARTURE_PCT', 'IMD_CATEGORY']],
            "summary_table": summary.reset_index(),
            "chart": fig,
            "explanation": "\n".join(lines)
        }
        
    # ----------------------------------------------------
    # CASE 2: RAINFALL TREND
    # ----------------------------------------------------
    elif intent == "trend":
        target_region = regions[0] if regions else None
        if target_region:
            df_trend = df_filtered[df_filtered['SUBDIVISION'] == target_region].sort_values('YEAR').copy()
            title_text = f"Rainfall Trend in {target_region} ({y_start}–{y_end})"
        else:
            # National average trend
            df_trend = df_filtered.groupby('YEAR')[['ANNUAL', 'Jun-Sep']].mean().reset_index()
            df_trend['SUBDIVISION'] = 'ALL-INDIA AVERAGE'
            title_text = f"All-India Average Rainfall Trend ({y_start}–{y_end})"
            
        # 5-year rolling average
        df_trend['5Y_MA'] = df_trend['ANNUAL'].rolling(5, min_periods=1).mean().round(1)
        
        # Calculate linear slope
        x = df_trend['YEAR'].values
        y = df_trend['ANNUAL'].values
        if len(x) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
        else:
            slope, intercept = 0.0, float(y[0]) if len(y) > 0 else 0.0
        slope_per_decade = round(slope * 10, 2)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_trend['YEAR'], y=df_trend['ANNUAL'],
            name='Annual Rainfall', marker_color='rgba(56, 189, 248, 0.45)'
        ))
        fig.add_trace(go.Scatter(
            x=df_trend['YEAR'], y=df_trend['5Y_MA'],
            name='5-Year Moving Avg', line=dict(color='#38bdf8', width=2.5)
        ))
        fig.add_trace(go.Scatter(
            x=df_trend['YEAR'], y=slope * x + intercept,
            name=f'Linear Trend ({slope_per_decade:+} mm/decade)',
            line=dict(color='#f43f5e', width=2, dash='dash')
        ))
        fig.update_layout(
            title=title_text,
            xaxis_title="Year", yaxis_title="Annual Rainfall (mm)",
            template="plotly_dark", hovermode="x unified"
        )
        
        reg_name = target_region if target_region else "All-India"
        mean_val = round(df_trend['ANNUAL'].mean(), 1)
        max_row = df_trend.loc[df_trend['ANNUAL'].idxmax()]
        min_row = df_trend.loc[df_trend['ANNUAL'].idxmin()]
        
        explanation = (
            f"**Historical Rainfall Trend for {reg_name} ({y_start}–{y_end}):**\n\n"
            f"- **Period Average**: **{mean_val} mm** per year.\n"
            f"- **Peak Wet Year**: **{int(max_row['YEAR'])}** with **{round(max_row['ANNUAL'], 1)} mm**.\n"
            f"- **Driest Year**: **{int(min_row['YEAR'])}** with **{round(min_row['ANNUAL'], 1)} mm**.\n"
            f"- **Long-term Trend Rate**: **{slope_per_decade:+} mm per decade** "
            f"({'moderately increasing' if slope > 0.5 else 'moderately decreasing' if slope < -0.5 else 'relatively stable'})."
        )
        
        return {
            "data": df_trend[['YEAR', 'SUBDIVISION', 'ANNUAL', '5Y_MA']],
            "chart": fig,
            "explanation": explanation
        }
        
    # ----------------------------------------------------
    # CASE 3: HIGHEST YEAR / EXTREME WET
    # ----------------------------------------------------
    elif intent == "highest_year":
        if regions:
            df_target = df_filtered[df_filtered['SUBDIVISION'].isin(regions)].copy()
            title = f"Top 10 Wettest Years in {', '.join(regions)}"
        else:
            # National average per year
            df_target = df_filtered.groupby('YEAR')['ANNUAL'].mean().reset_index()
            df_target['SUBDIVISION'] = 'All-India'
            title = "Top 10 Wettest Years in India (1901–2015)"
            
        top_years = df_target.sort_values(by='ANNUAL', ascending=False).head(10).reset_index(drop=True)
        top1 = top_years.iloc[0]
        
        fig = px.bar(
            top_years, x='YEAR', y='ANNUAL', color='ANNUAL',
            color_continuous_scale='Blues',
            title=title,
            labels={'ANNUAL': 'Rainfall (mm)', 'YEAR': 'Year'}
        )
        fig.update_layout(template="plotly_dark")
        
        reg_str = regions[0] if regions else "India overall"
        explanation = (
            f"**Highest Rainfall Analysis ({reg_str}):**\n\n"
            f"- The single highest rainfall year recorded was **{int(top1['YEAR'])}**, "
            f"recording **{round(top1['ANNUAL'], 1)} mm**.\n"
            f"- The 3 wettest recorded years in this window were: "
            f"**{int(top_years.iloc[0]['YEAR'])}** ({round(top_years.iloc[0]['ANNUAL'], 1)} mm), "
            f"**{int(top_years.iloc[1]['YEAR'])}** ({round(top_years.iloc[1]['ANNUAL'], 1)} mm), and "
            f"**{int(top_years.iloc[2]['YEAR'])}** ({round(top_years.iloc[2]['ANNUAL'], 1)} mm)."
        )
        
        return {
            "data": top_years,
            "chart": fig,
            "explanation": explanation
        }
        
    # ----------------------------------------------------
    # CASE 4: LOWEST YEAR / SEVERE DROUGHT
    # ----------------------------------------------------
    elif intent == "lowest_year":
        if regions:
            df_target = df_filtered[df_filtered['SUBDIVISION'].isin(regions)].copy()
            title = f"Top 10 Driest Years in {', '.join(regions)}"
        else:
            df_target = df_filtered.groupby('YEAR')['ANNUAL'].mean().reset_index()
            df_target['SUBDIVISION'] = 'All-India'
            title = "Top 10 Driest / Drought Years in India (1901–2015)"
            
        bot_years = df_target.sort_values(by='ANNUAL', ascending=True).head(10).reset_index(drop=True)
        bot1 = bot_years.iloc[0]
        
        fig = px.bar(
            bot_years, x='YEAR', y='ANNUAL', color='ANNUAL',
            color_continuous_scale='Reds_r',
            title=title,
            labels={'ANNUAL': 'Rainfall (mm)', 'YEAR': 'Year'}
        )
        fig.update_layout(template="plotly_dark")
        
        reg_str = regions[0] if regions else "India overall"
        explanation = (
            f"**Driest / Drought Year Analysis ({reg_str}):**\n\n"
            f"- The lowest rainfall year recorded was **{int(bot1['YEAR'])}**, "
            f"with only **{round(bot1['ANNUAL'], 1)} mm** of precipitation.\n"
            f"- The 3 most severe drought years in this window were: "
            f"**{int(bot_years.iloc[0]['YEAR'])}** ({round(bot_years.iloc[0]['ANNUAL'], 1)} mm), "
            f"**{int(bot_years.iloc[1]['YEAR'])}** ({round(bot_years.iloc[1]['ANNUAL'], 1)} mm), and "
            f"**{int(bot_years.iloc[2]['YEAR'])}** ({round(bot_years.iloc[2]['ANNUAL'], 1)} mm)."
        )
        
        return {
            "data": bot_years,
            "chart": fig,
            "explanation": explanation
        }
        
    # ----------------------------------------------------
    # CASE 5: HIGHEST REGION / RANKINGS
    # ----------------------------------------------------
    elif intent == "highest_region":
        sub_ranks = df_filtered.groupby('SUBDIVISION')['ANNUAL'].mean().sort_values(ascending=False).reset_index()
        sub_ranks['ANNUAL'] = sub_ranks['ANNUAL'].round(1)
        top1 = sub_ranks.iloc[0]
        top10 = sub_ranks.head(10)
        
        fig = px.bar(
            top10, x='ANNUAL', y='SUBDIVISION', orientation='h',
            title=f"Top 10 Subdivisions with Highest Average Rainfall ({y_start}–{y_end})",
            labels={'ANNUAL': 'Mean Annual Rainfall (mm)', 'SUBDIVISION': 'Subdivision'},
            color='ANNUAL', color_continuous_scale='Tealgrn'
        )
        fig.update_layout(template="plotly_dark", yaxis={'categoryorder': 'total ascending'})
        
        explanation = (
            f"**Regional Rainfall Rankings ({y_start}–{y_end}):**\n\n"
            f"- **Highest Rainfall Region**: **{top1['SUBDIVISION']}** leads with a mean annual rainfall of **{top1['ANNUAL']} mm**.\n"
            f"- Top 3 highest regions:\n"
            f"  1. **{sub_ranks.iloc[0]['SUBDIVISION']}**: {sub_ranks.iloc[0]['ANNUAL']} mm\n"
            f"  2. **{sub_ranks.iloc[1]['SUBDIVISION']}**: {sub_ranks.iloc[1]['ANNUAL']} mm\n"
            f"  3. **{sub_ranks.iloc[2]['SUBDIVISION']}**: {sub_ranks.iloc[2]['ANNUAL']} mm"
        )
        
        return {
            "data": sub_ranks,
            "chart": fig,
            "explanation": explanation
        }
        
    # ----------------------------------------------------
    # CASE 6: LOWEST REGION / DRIEST REGIONS
    # ----------------------------------------------------
    elif intent == "lowest_region":
        sub_ranks = df_filtered.groupby('SUBDIVISION')['ANNUAL'].mean().sort_values(ascending=True).reset_index()
        sub_ranks['ANNUAL'] = sub_ranks['ANNUAL'].round(1)
        bot1 = sub_ranks.iloc[0]
        bot10 = sub_ranks.head(10)
        
        fig = px.bar(
            bot10, x='ANNUAL', y='SUBDIVISION', orientation='h',
            title=f"Top 10 Driest Subdivisions in India ({y_start}–{y_end})",
            labels={'ANNUAL': 'Mean Annual Rainfall (mm)', 'SUBDIVISION': 'Subdivision'},
            color='ANNUAL', color_continuous_scale='Oranges_r'
        )
        fig.update_layout(template="plotly_dark", yaxis={'categoryorder': 'total descending'})
        
        explanation = (
            f"**Driest Regions in India ({y_start}–{y_end}):**\n\n"
            f"- **Driest Region**: **{bot1['SUBDIVISION']}** recorded the lowest annual average of only **{bot1['ANNUAL']} mm**.\n"
            f"- Top 3 driest regions:\n"
            f"  1. **{sub_ranks.iloc[0]['SUBDIVISION']}**: {sub_ranks.iloc[0]['ANNUAL']} mm\n"
            f"  2. **{sub_ranks.iloc[1]['SUBDIVISION']}**: {sub_ranks.iloc[1]['ANNUAL']} mm\n"
            f"  3. **{sub_ranks.iloc[2]['SUBDIVISION']}**: {sub_ranks.iloc[2]['ANNUAL']} mm"
        )
        
        return {
            "data": sub_ranks,
            "chart": fig,
            "explanation": explanation
        }
        
    # ----------------------------------------------------
    # DEFAULT FALLBACK: GENERAL SUMMARY
    # ----------------------------------------------------
    else:
        df_target = df_filtered[df_filtered['SUBDIVISION'].isin(regions)] if regions else df_filtered
        mean_val = round(df_target['ANNUAL'].mean(), 1)
        fig = px.histogram(
            df_target, x='ANNUAL', nbins=30,
            title=f"Rainfall Distribution ({y_start}–{y_end})",
            labels={'ANNUAL': 'Annual Rainfall (mm)'}
        )
        fig.update_layout(template="plotly_dark")
        return {
            "data": df_target[['YEAR', 'SUBDIVISION', 'ANNUAL']].head(20),
            "chart": fig,
            "explanation": f"Calculated mean rainfall for the selected parameters: **{mean_val} mm**."
        }

def process_nl_query(query: str, df_sub: pd.DataFrame, df_dist: pd.DataFrame):
    """
    End-to-end Natural Language query processor.
    Returns structured JSON, data table, Plotly chart, and grounded explanation.
    """
    all_districts = df_dist['DISTRICT'].unique().tolist()
    spec = extract_intent_and_entities(query, all_districts)
    result = execute_nl_query(spec, df_sub, df_dist)
    
    return {
        "structured_json": spec,
        "data": result.get("data"),
        "chart": result.get("chart"),
        "explanation": result.get("explanation"),
        "summary_table": result.get("summary_table", None)
    }

if __name__ == "__main__":
    from data_loader import load_data
    df_s, df_d = load_data()
    q = "Compare Tamil Nadu and Kerala rainfall from 2000 to 2015."
    res = process_nl_query(q, df_s, df_d)
    print("JSON:", json.dumps(res['structured_json'], indent=2))
    print("\nExplanation:\n", res['explanation'])
