import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import datetime
import calendar

# --- Page Config ---
st.set_page_config(page_title="Punjab Wheat Input Cost Forecaster", layout="wide")

# --- Custom CSS Theme ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,700&family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;600&display=swap');

h1, h2, h3 {
    font-family: 'Fraunces', serif !important;
}

h1 {
    font-size: 2.5rem !important; /* ~40px */
}

h2 {
    font-size: 2rem !important; /* ~32px */
}

h3 {
    font-size: 1.5rem !important; /* ~24px */
}

body, p, label, .stMarkdown {
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
}

[data-testid="stSidebar"] {
    background-color: #1A2C42 !important;
    border-left: 2px solid #E0A83E !important;
}

.caption-text {
    color: #8A97AC !important;
    font-family: 'Inter', sans-serif !important;
}

.gold-divider {
    border-top: 1px solid #E0A83E;
    margin: 20px 0;
}

.eyebrow {
    color: #7C8BA1;
    letter-spacing: 2px;
    font-size: 0.85rem !important; /* ~14px */
    font-weight: 600;
    text-transform: uppercase;
    font-family: 'Inter', sans-serif;
    margin-bottom: -15px;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="eyebrow">GOVERNMENT DATA · CACP · PUNJAB WHEAT · 2017–2022</div>', unsafe_allow_html=True)
st.title("Punjab Wheat Input Cost Forecaster")
st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

st.markdown("""
Welcome to the Punjab Wheat Input Cost Forecaster. This application forecasts agricultural input costs 
(Seed, Fertilizer, Irrigation, Human Labour, and Total Cost of Cultivation) for wheat farming in Punjab. 
It is based on official CACP (Commission for Agricultural Costs & Prices) data from 2017-18 to 2021-22. 
Use this tool to help farmers and agri-businesses with financial planning.
""")

# --- Load Data and Models ---
@st.cache_data
def load_historical_data():
    return pd.read_csv("punjab_wheat_cleaned_wide.csv")

@st.cache_resource
def load_models():
    models = {
        'Total Cost of Cultivation C2 (Rs./Hectare)': joblib.load('models/model_total_cost_of_cultivation_c2.pkl'),
        'Fertilizer cost (Rs./Hectare)': joblib.load('models/model_fertilizer_cost.pkl'),
        'Irrigation charges (Rs./Hectare)': joblib.load('models/model_irrigation_charges.pkl'),
        'Seed cost (Rs./Hectare)': joblib.load('models/model_seed_cost.pkl'),
        'Human Labour cost (Rs./Hectare)': joblib.load('models/model_human_labour_cost.pkl')
    }
    return models

df_historical = load_historical_data()
models = load_models()

# --- Sidebar / User Input ---
st.sidebar.header("Forecast Settings")
target_year = st.sidebar.slider("Select Year to Forecast", min_value=2022, max_value=2027, value=2022, step=1)
generate_btn = st.sidebar.button("Generate Forecast")

# --- Main Logic ---
if generate_btn:
    st.header(f"Forecast for Year: {target_year}-{str(target_year+1)[-2:]}")
    
    # 1. Predictions
    time_index_val = target_year - 2017
    X_future = pd.DataFrame({'time_index': [time_index_val]})
    
    predictions = {}
    for target_name, model in models.items():
        predictions[target_name] = model.predict(X_future)[0]
    
    # 2. Display Metrics
    st.subheader("Predicted Costs (Rs./Hectare)")
    cols = st.columns(len(predictions))
    
    for idx, (target_name, pred_val) in enumerate(predictions.items()):
        short_name = target_name.split(' (')[0].replace('Total Cost of Cultivation C2', 'Total Cost (C2)')
        cols[idx].metric(label=short_name, value=f"₹{pred_val:,.2f}")
    
    # 3. Bar Chart of Predicted Components
    st.subheader("Cost Components Breakdown")
    components = {k: v for k, v in predictions.items() if 'Total' not in k}
    
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0F1C2E')
    ax.set_facecolor('#0F1C2E')
    
    # Earthy palette adapted for navy background
    earthy_palette = ['#8FA888', '#C97C7C', '#6BA3A0', '#B89B7A']
    bars = ax.bar([k.split(' (')[0] for k in components.keys()], components.values(), color=earthy_palette)
    
    ax.set_ylabel("Cost (Rs./Hectare)", color='#F2ECE1')
    ax.set_title(f"Predicted Input Costs for {target_year}-{str(target_year+1)[-2:]} (Excluding Total C2)", color='#F2ECE1')
    ax.tick_params(colors='#F2ECE1')
    for spine in ax.spines.values():
        spine.set_edgecolor('#8A97AC')
        
    ax.set_ylim(0, max(components.values()) * 1.2)
    # Add values on top
    for bar in bars:
        yval = bar.get_height()
        ax.annotate(f'₹{yval:,.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, yval),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=13, fontweight='bold', color='#F2ECE1',
                    bbox=dict(facecolor='#1A2C42', alpha=0.85, edgecolor='#E0A83E', linewidth=1, pad=3))
    st.pyplot(fig)
    
    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)
    
    # 4. Explainability (SHAP)
    st.subheader("Why these numbers? (Trend Explanation)")
    st.write("Using SHAP (SHapley Additive exPlanations), we can see how the historical time trend contributes to these predictions.")
    
    # We train explainers on historical X
    df_historical['time_index'] = df_historical['Year_Num'] - df_historical['Year_Num'].min()
    X_historical = df_historical[['time_index']]
    
    explain_cols = st.columns(2)
    for i, (target_name, model) in enumerate(models.items()):
        short_name = target_name.split(' (')[0].replace('Total Cost of Cultivation C2', 'Total Cost (C2)')
        
        explainer = shap.LinearExplainer(model, X_historical)
        shap_values = explainer.shap_values(X_future)
        impact = shap_values[0][0]
        trend_coef = model.coef_[0]
        
        with explain_cols[i % 2]:
            st.markdown(f"**{short_name}**: Projected to **{'rise' if trend_coef > 0 else 'fall'}** by approximately **₹{abs(trend_coef):.2f} per year**.")
            st.caption(f"For {target_year}, the time trend alone pushes this cost {'up' if impact > 0 else 'down'} by ₹{abs(impact):,.2f} relative to the historical 5-year average.")
    
    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)
    
    # 5. Historical Context Line Chart
    st.subheader("Historical Context & Forecast Trend")
    
    # Combine historical and forecasted data for plotting
    df_plot = df_historical[['Year_Num'] + list(predictions.keys())].copy()
    
    new_row = {'Year_Num': target_year}
    for k, v in predictions.items():
        new_row[k] = v
        
    df_plot = pd.concat([df_plot, pd.DataFrame([new_row])], ignore_index=True)
    df_plot.sort_values('Year_Num', inplace=True)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.patch.set_facecolor('#0F1C2E')
    ax2.set_facecolor('#0F1C2E')
    
    ax2.set_xlabel("Year", color='#F2ECE1')
    ax2.set_ylabel("Cost (Rs./Hectare)", color='#F2ECE1')
    ax2.tick_params(colors='#F2ECE1')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#8A97AC')

    color_map = {
        'Total Cost (C2)': '#E0A83E',
        'Fertilizer cost': '#8FA888',
        'Irrigation charges': '#C97C7C',
        'Seed cost': '#6BA3A0',
        'Human Labour cost': '#B89B7A'
    }

    for target_name in predictions.keys():
        short_name = target_name.split(' (')[0].replace('Total Cost of Cultivation C2', 'Total Cost (C2)')
        line_color = color_map.get(short_name, '#F2ECE1')
        
        # Plot historical
        historical_part = df_plot[df_plot['Year_Num'] <= 2021]
        ax2.plot(historical_part['Year_Num'], historical_part[target_name], marker='o', label=short_name, color=line_color)
        
        # Plot forecast (dashed line from last historical point)
        forecast_part = df_plot[df_plot['Year_Num'] >= 2021]
        ax2.plot(forecast_part['Year_Num'], forecast_part[target_name], linestyle='--', color=line_color)
        
        # Mark forecast point
        marker_size = 250 if 'Total Cost' in short_name else 100
        ax2.scatter(target_year, predictions[target_name], color=line_color, marker='*', s=marker_size, zorder=5)
        
        # Only add value labels for Total Cost (C2) to avoid clutter
        if 'Total Cost' in short_name:
            # Label historical points
            for i, (_, row) in enumerate(historical_part.iterrows()):
                y_offset = 20 if i % 2 == 1 else 8
                ax2.annotate(f"₹{row[target_name]:,.0f}", 
                             xy=(row['Year_Num'], row[target_name]),
                             xytext=(0, y_offset), textcoords='offset points',
                             ha='center', va='bottom',
                             fontsize=11, fontweight='bold', color='#F2ECE1',
                             bbox=dict(facecolor='#1A2C42', alpha=0.85, edgecolor='#E0A83E', linewidth=1, pad=3))
            # Label forecast point
            forecast_index = len(historical_part)
            y_offset_forecast = 20 if forecast_index % 2 == 1 else 8
            ax2.annotate(f"₹{predictions[target_name]:,.0f}", 
                         xy=(target_year, predictions[target_name]),
                         xytext=(0, y_offset_forecast), textcoords='offset points',
                         ha='center', va='bottom',
                         fontsize=11, fontweight='bold', color='#F2ECE1',
                         bbox=dict(facecolor='#1A2C42', alpha=0.85, edgecolor='#E0A83E', linewidth=1, pad=3))
    
    ax2.set_xticks(df_plot['Year_Num'].unique())
    # Format x-ticks as Year-Year+1
    ax2.set_xticklabels([f"{int(y)}-{str(int(y)+1)[-2:]}" for y in df_plot['Year_Num'].unique()])
    legend = ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', facecolor='#1A2C42', edgecolor='#8A97AC')
    for text in legend.get_texts():
        text.set_color('#F2ECE1')
    ax2.grid(True, linestyle='--', alpha=0.3, color='#8A97AC')
    
    st.pyplot(fig2)

    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)
    
    # 6. Seasonal Planning Calendar
    st.subheader("Seasonal Planning Calendar")
    
    months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']
    
    seed_pred = predictions.get('Seed cost (Rs./Hectare)', 0)
    fert_pred = predictions.get('Fertilizer cost (Rs./Hectare)', 0)
    irr_pred = predictions.get('Irrigation charges (Rs./Hectare)', 0)
    labour_pred = predictions.get('Human Labour cost (Rs./Hectare)', 0)
    total_c2_pred = predictions.get('Total Cost of Cultivation C2 (Rs./Hectare)', 0)
    
    allocations = {
        'Seed cost': [0, seed_pred * 1.0, 0, 0, 0, 0, 0],
        'Fertilizer cost': [0, fert_pred * 0.6, fert_pred * 0.4, 0, 0, 0, 0],
        'Irrigation charges': [0, 0, irr_pred * 0.25, irr_pred * 0.25, irr_pred * 0.25, irr_pred * 0.125, irr_pred * 0.125],
        'Human Labour cost': [0, labour_pred * 0.3, 0, labour_pred * 0.2, 0, labour_pred * 0.25, labour_pred * 0.25]
    }
    
    unallocated = total_c2_pred - (seed_pred + fert_pred + irr_pred + labour_pred)
    allocations['Total Cost (C2)'] = [(unallocated / 7) + sum(allocations[k][i] for k in ['Seed cost', 'Fertilizer cost', 'Irrigation charges', 'Human Labour cost']) for i in range(7)]

    # --- Custom Date Range Estimator ---
    st.markdown("#### Custom Date Range Estimator")
    st.caption("Estimated costs for your selected period, prorated from the seasonal crop-calendar allocation shown below. Not based on measured daily/weekly data — CACP publishes cost data annually only.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        granularity = st.radio("View granularity", ["Day-to-day", "Week-to-week", "Month-to-month"], horizontal=True)
    with col2:
        season_start = datetime.date(target_year, 10, 1)
        season_end = datetime.date(target_year + 1, 4, 30)
        
        date_range = st.date_input("Select Date Range (Oct 1 - Apr 30)", 
                                   value=(season_start, season_end), 
                                   min_value=season_start, 
                                   max_value=season_end)
                                   
    if isinstance(date_range, tuple) and len(date_range) == 2:
        sel_start, sel_end = date_range
        days_in_range = (sel_end - sel_start).days + 1
        
        if days_in_range > 0:
            month_info = [
                (10, target_year, 'Oct'),
                (11, target_year, 'Nov'),
                (12, target_year, 'Dec'),
                (1, target_year + 1, 'Jan'),
                (2, target_year + 1, 'Feb'),
                (3, target_year + 1, 'Mar'),
                (4, target_year + 1, 'Apr')
            ]
            
            dates_idx = pd.date_range(sel_start, sel_end)
            daily_costs = {k: np.zeros(len(dates_idx)) for k in allocations.keys()}
            
            for i, (m, y, m_name) in enumerate(month_info):
                days_in_m = calendar.monthrange(y, m)[1]
                m_start = datetime.date(y, m, 1)
                m_end = datetime.date(y, m, days_in_m)
                
                overlap = (dates_idx.date >= m_start) & (dates_idx.date <= m_end)
                if overlap.any():
                    for cat, alloc in allocations.items():
                        daily_rate = alloc[i] / days_in_m
                        daily_costs[cat][overlap] = daily_rate
                        
            df_daily = pd.DataFrame(daily_costs, index=dates_idx)
            
            st.markdown(f"**Total Estimated Cost for Selected Window ({days_in_range} days):**")
            
            if granularity == "Day-to-day":
                cols_day = st.columns(5)
                for j, cat in enumerate(allocations.keys()):
                    tot = df_daily[cat].sum()
                    rate = tot / days_in_range
                    cols_day[j].metric(cat, f"₹{tot:,.0f}", f"₹{rate:,.0f}/day", delta_color="off")
                    
            elif granularity == "Week-to-week":
                df_weekly = df_daily.resample('W').sum()
                df_weekly.index = df_weekly.index.strftime('%b %d, %Y')
                st.bar_chart(df_weekly, color=[color_map.get(c, '#8A97AC') for c in df_weekly.columns])
                
            elif granularity == "Month-to-month":
                df_monthly = df_daily.resample('ME').sum()
                df_monthly.index = df_monthly.index.strftime('%b %Y')
                st.bar_chart(df_monthly, color=[color_map.get(c, '#8A97AC') for c in df_monthly.columns])
        
    st.markdown("<hr class='gold-divider' style='border-top: 1px dashed #E0A83E; opacity: 0.5;'>", unsafe_allow_html=True)
    
    # Component 1 - When to Budget
    st.markdown("#### When to Budget: Wheat Crop Calendar (Punjab)")
    st.caption("This shows WHEN during the season each cost is typically incurred, based on Punjab Agricultural University (PAU) recommended practices — not measured monthly data.")
    
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    fig3.patch.set_facecolor('#0F1C2E')
    ax3.set_facecolor('#0F1C2E')
    
    bottoms = np.zeros(len(months))
    for component, alloc in allocations.items():
        if component == 'Total Cost (C2)':
            continue
        color = color_map.get(component, '#F2ECE1')
        ax3.bar(months, alloc, bottom=bottoms, label=component, color=color)
        bottoms += np.array(alloc)
        
    ax3.set_ylabel("Estimated Cost (Rs./Hectare)", color='#F2ECE1')
    ax3.tick_params(colors='#F2ECE1')
    for spine in ax3.spines.values():
        spine.set_edgecolor('#8A97AC')
        
    legend3 = ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', facecolor='#1A2C42', edgecolor='#8A97AC')
    for text in legend3.get_texts():
        text.set_color('#F2ECE1')
    
    st.pyplot(fig3)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Component 2 - Climate Context
    st.markdown("#### Climate Context for Punjab (Typical Monthly Averages)")
    st.caption("Typical monthly climate normals for Punjab — shown for seasonal planning context. These are historical averages, not a forecast, and are not statistically linked to the cost model above.")
    
    climate_data = pd.DataFrame({
        'Month': ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr'],
        'Temperature (°C)': [25, 19, 14, 13, 16, 21, 27],
        'Rainfall (mm)': [15, 5, 15, 25, 30, 20, 10]
    })
    
    fig4, ax4_temp = plt.subplots(figsize=(10, 4))
    fig4.patch.set_facecolor('#0F1C2E')
    ax4_temp.set_facecolor('#0F1C2E')
    
    ax4_temp.plot(climate_data['Month'], climate_data['Temperature (°C)'], color='#E0A83E', marker='o', label='Temperature (°C)', linewidth=2)
    ax4_temp.set_ylabel("Temperature (°C)", color='#E0A83E')
    ax4_temp.tick_params(axis='y', colors='#E0A83E')
    ax4_temp.tick_params(axis='x', colors='#F2ECE1')
    
    ax4_rain = ax4_temp.twinx()
    ax4_rain.plot(climate_data['Month'], climate_data['Rainfall (mm)'], color='#6BA3A0', marker='s', linestyle='--', label='Rainfall (mm)', linewidth=2)
    ax4_rain.set_ylabel("Rainfall (mm)", color='#6BA3A0')
    ax4_rain.tick_params(axis='y', colors='#6BA3A0')
    
    for ax in [ax4_temp, ax4_rain]:
        for spine in ax.spines.values():
            spine.set_edgecolor('#8A97AC')
            
    # Combine legends for twin axes
    lines_1, labels_1 = ax4_temp.get_legend_handles_labels()
    lines_2, labels_2 = ax4_rain.get_legend_handles_labels()
    fig4.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax4_temp.transAxes, facecolor='#1A2C42', edgecolor='#8A97AC', labelcolor='#F2ECE1')
    
    st.pyplot(fig4)
    
    st.markdown("""
    <div style="color: #F2ECE1; font-family: 'Inter', sans-serif; font-size: 0.9em; margin-top: 10px;">
    <strong>Context:</strong> Low rainfall during November through February means that irrigation is essential (not optional) during this critical growth window. This explains why irrigation charges cluster heavily in these winter months.
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)
# --- Footer / Disclaimer ---
st.markdown("""
<div class="caption-text">
<b>Disclaimer</b>: Data source is CACP/Ministry of Agriculture (2017-18 to 2021-22). 
Forecasts are generated using a simple linear trend model trained on limited historical data (5 years). 
This tool should be used strictly as a planning aid and does not constitute a financial guarantee.
</div>
""", unsafe_allow_html=True)
