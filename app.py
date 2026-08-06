import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# --- Page Config ---
st.set_page_config(page_title="Punjab Wheat Input Cost Forecaster", layout="wide")

# --- Custom CSS Theme ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,700&family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;600&display=swap');

h1, h2, h3 {
    font-family: 'Fraunces', serif !important;
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
    font-size: 0.8em;
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
                    fontsize=13, fontweight='bold', color='#0F1C2E',
                    bbox=dict(facecolor='#F2ECE1', alpha=0.9, edgecolor='none', pad=2))
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
                             fontsize=11, fontweight='bold', color='#0F1C2E',
                             bbox=dict(facecolor='#F2ECE1', alpha=0.9, edgecolor='none', pad=1))
            # Label forecast point
            forecast_index = len(historical_part)
            y_offset_forecast = 20 if forecast_index % 2 == 1 else 8
            ax2.annotate(f"₹{predictions[target_name]:,.0f}", 
                         xy=(target_year, predictions[target_name]),
                         xytext=(0, y_offset_forecast), textcoords='offset points',
                         ha='center', va='bottom',
                         fontsize=11, fontweight='bold', color='#0F1C2E',
                         bbox=dict(facecolor='#F2ECE1', alpha=0.9, edgecolor='none', pad=1))
    
    ax2.set_xticks(df_plot['Year_Num'].unique())
    # Format x-ticks as Year-Year+1
    ax2.set_xticklabels([f"{int(y)}-{str(int(y)+1)[-2:]}" for y in df_plot['Year_Num'].unique()])
    legend = ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', facecolor='#1A2C42', edgecolor='#8A97AC')
    for text in legend.get_texts():
        text.set_color('#F2ECE1')
    ax2.grid(True, linestyle='--', alpha=0.3, color='#8A97AC')
    
    st.pyplot(fig2)

st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)
# --- Footer / Disclaimer ---
st.markdown("""
<div class="caption-text">
<b>Disclaimer</b>: Data source is CACP/Ministry of Agriculture (2017-18 to 2021-22). 
Forecasts are generated using a simple linear trend model trained on limited historical data (5 years). 
This tool should be used strictly as a planning aid and does not constitute a financial guarantee.
</div>
""", unsafe_allow_html=True)
