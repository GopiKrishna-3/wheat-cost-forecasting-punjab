import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration
file_path = "punjab_wheat_input_costs_tidy_2017-2022.csv"
output_csv = "punjab_wheat_cleaned_wide.csv"

print("TASK 1: Loading and pivoting data...")
df = pd.read_csv(file_path)

# Pivot the data
df_wide = df.pivot(index='Year', columns='Metric', values='Value').reset_index()

# Convert "20XX-YY" to starting year integer
df_wide['Year_Num'] = df_wide['Year'].str.split('-').str[0].astype(int)

# Sort by year just in case
df_wide = df_wide.sort_values('Year_Num').reset_index(drop=True)


print("\nTASK 2: Filtering key metrics...")
# Define our target column mappings
mapping = {
    'Seed': 'Seed cost (Rs./Hectare)',
    'Fertilizer': 'Fertilizer cost (Rs./Hectare)',
    'Labour': 'Human Labour cost (Rs./Hectare)',
    'Irrigation': 'Irrigation charges (Rs./Hectare)',
    'C2': 'Total Cost of Cultivation C2 (Rs./Hectare)'
}

# Find the exact column names that match our partial text
matched_cols = {}
for col in df_wide.columns:
    if "Operational Cost > Seed" == col:
        matched_cols[col] = mapping['Seed']
    elif "Operational Cost > Fertilizer & Manure - Fertilizer" == col:
        matched_cols[col] = mapping['Fertilizer']
    elif "Operational Cost > Total" == col:
        matched_cols[col] = mapping['Labour']
    elif "Irrigation Charges" in col:
        matched_cols[col] = mapping['Irrigation']
    elif "Cost of Cultivation (Rs./Hectare) - C2" == col:
        matched_cols[col] = mapping['C2']

print("Matched columns found:")
for original, new in matched_cols.items():
    print(f"  '{original}' -> '{new}'")

# Check if we found all 5
if len(matched_cols) < 5:
    print(f"Warning: Expected 5 columns, found {len(matched_cols)}")

# Keep Year, Year_Num and the matched columns
keep_cols = ['Year', 'Year_Num'] + list(matched_cols.keys())
df_clean = df_wide[keep_cols].copy()

# Rename the columns
df_clean = df_clean.rename(columns=matched_cols)


print("\nTASK 3 & 4: Data Quality Checks and Cleaning...")
key_columns = list(matched_cols.values())

# Ensure all cost columns are numeric
for col in key_columns:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Check for missing values
missing_counts = df_clean.isnull().sum()
if missing_counts.sum() > 0:
    print("Warning: Missing values found!")
    print(missing_counts[missing_counts > 0])
else:
    print("No missing values found in key columns.")

print("\nSummary Statistics (Min, Max, Mean):")
summary = df_clean[key_columns].agg(['min', 'max', 'mean'])
print(summary)


print("\nTASK 5: EDA Visualizations...")
# (a) Line chart showing all 5 key cost metrics
plt.figure(figsize=(10, 6))
for col in key_columns:
    plt.plot(df_clean['Year_Num'], df_clean[col], marker='o', label=col)
    
plt.title('Key Cost Metrics for Wheat Cultivation in Punjab (2017-2022)')
plt.xlabel('Year')
plt.ylabel('Cost (Rs./Hectare)')
plt.xticks(df_clean['Year_Num'], df_clean['Year'])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('cost_metrics_trend.png', bbox_inches='tight')
plt.close()
print("Saved line chart to 'cost_metrics_trend.png'")

# (b) Bar chart showing YoY % change in Total Cost of Cultivation (C2)
plt.figure(figsize=(8, 5))
c2_col = mapping['C2']
df_clean['C2_YoY_Change'] = df_clean[c2_col].pct_change() * 100

# Drop the first year (NaN) for plotting the changes
plot_data = df_clean.dropna(subset=['C2_YoY_Change'])
bars = plt.bar(plot_data['Year'], plot_data['C2_YoY_Change'], color='skyblue', edgecolor='black')

plt.title('Year-over-Year % Change in Total Cost of Cultivation (C2)')
plt.xlabel('Year')
plt.ylabel('% Change')
plt.axhline(0, color='black', linewidth=1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + (0.5 if yval > 0 else -1.5), 
             f'{yval:.1f}%', ha='center', va='bottom' if yval > 0 else 'top')

plt.tight_layout()
plt.savefig('c2_yoy_change.png')
plt.close()
print("Saved bar chart to 'c2_yoy_change.png'")

# Save cleaned CSV
df_clean.to_csv(output_csv, index=False)
print(f"\nSaved cleaned wide dataset to '{output_csv}'")
