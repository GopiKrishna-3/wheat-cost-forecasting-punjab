import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import shap

print("--- AI-Based Agricultural Input Cost Forecasting ---")

# Setup
file_path = "punjab_wheat_cleaned_wide.csv"
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# Define our 5 target columns based on the CSV
targets = [
    'Total Cost of Cultivation C2 (Rs./Hectare)',
    'Fertilizer cost (Rs./Hectare)',
    'Irrigation charges (Rs./Hectare)',
    'Seed cost (Rs./Hectare)',
    'Human Labour cost (Rs./Hectare)'
]

print("\nTASK 1: Loading data and Feature Engineering...")
df = pd.read_csv(file_path)
# Feature 1: Year_Num (2017, 2018, etc.)
# Feature 2: time_index (0, 1, 2, 3, 4) to handle scale issues
df['time_index'] = df['Year_Num'] - df['Year_Num'].min()

# We will use 'time_index' as our primary X to ensure numerical stability in linear models
# compared to using large numbers like 2020 which can sometimes distort coefficients.
X = df[['time_index']]
print("Using 'time_index' (0, 1, 2, 3, 4) as our primary feature X representing years 2017 to 2021.")


print("\nTASK 2 & 3: Model Training and Evaluation (LOOCV & Train/Test Split)...")
# Why Linear Regression?
print("Note on Model Selection: We are using a simple Linear Regression model.")
print("Given we only have 5 data points, complex models like Random Forests or Neural Networks ")
print("would severely overfit (they would just memorize the 5 points). A linear model ")
print("enforces a simple trend line, which is statistically appropriate for this tiny sample size.")

loocv = LeaveOneOut()
eval_results = []

for target in targets:
    y = df[target]
    
    # --- LOOCV Evaluation ---
    loocv_maes = []
    loocv_rmses = []
    
    for train_index, test_index in loocv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        loocv_maes.append(mean_absolute_error(y_test, preds))
        loocv_rmses.append(np.sqrt(mean_squared_error(y_test, preds)))
        
    avg_mae_loocv = np.mean(loocv_maes)
    avg_rmse_loocv = np.mean(loocv_rmses)
    
    # --- Simple Train/Test Split (Train: 2017-2020, Test: 2021) ---
    X_train_split = X.iloc[:4]
    y_train_split = y.iloc[:4]
    X_test_split = X.iloc[4:5]
    y_test_split = y.iloc[4:5]
    
    model_split = LinearRegression()
    model_split.fit(X_train_split, y_train_split)
    preds_split = model_split.predict(X_test_split)
    
    mae_split = mean_absolute_error(y_test_split, preds_split)
    rmse_split = np.sqrt(mean_squared_error(y_test_split, preds_split))
    
    eval_results.append({
        'Target': target,
        'LOOCV_MAE': avg_mae_loocv,
        'LOOCV_RMSE': avg_rmse_loocv,
        'TrainTest_MAE': mae_split,
        'TrainTest_RMSE': rmse_split
    })

# Save and print evaluation
df_eval = pd.DataFrame(eval_results)
df_eval.to_csv("model_evaluation.csv", index=False)
print("\nEvaluation Results (LOOCV is the reliable metric here):")
print(df_eval.to_string(index=False))


print("\nTASK 4: Final Model Training & Forecasting...")
# We forecast for next 2 years (2022-23 and 2023-24).
# 2021-22 corresponds to Year_Num = 2021 and time_index = 4
# So 2022-23 is time_index = 5, and 2023-24 is time_index = 6
X_future = pd.DataFrame({'time_index': [5, 6]})
future_years = ['2022-23', '2023-24']
forecasts = {'Year': future_years, 'Year_Num': [2022, 2023]}
final_models = {}

for target in targets:
    y = df[target]
    # Retrain on ALL 5 years
    model_final = LinearRegression()
    model_final.fit(X, y)
    final_models[target] = model_final
    
    # Forecast
    preds_future = model_final.predict(X_future)
    forecasts[target] = preds_future

df_forecast = pd.DataFrame(forecasts)
df_forecast.to_csv("forecast_results.csv", index=False)
print("\nForecasts for 2022-23 and 2023-24:")
print(df_forecast.to_string(index=False))


print("\nTASK 5: Explainability (SHAP)...")
print("SHAP Summary: Because we are using a single-feature Linear Regression model,")
print("the SHAP value for 'time_index' simply represents the model's coefficient multiplied")
print("by the feature value relative to its expected (mean) value. A positive SHAP value")
print("means the passage of time is pushing the cost UP relative to the 5-year average.")
print("The magnitude shows exactly how many Rs./Hectare are added or subtracted per year.")

for target in targets:
    model = final_models[target]
    
    # SHAP LinearExplainer
    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)
    
    # We look at the SHAP value for the most recent year (time_index = 4)
    # to see how much 'time' influenced the latest cost
    latest_shap = shap_values[4][0]
    
    print(f"\nTarget: {target}")
    print(f" - Trend coefficient: {model.coef_[0]:.2f} Rs./Hectare per year")
    print(f" - SHAP value for 2021-22: {latest_shap:.2f} Rs./Hectare")
    if latest_shap > 0:
        print("   (This means time trend pushed the 2021-22 cost ABOVE the historical average.)")
    else:
        print("   (This means time trend pushed the 2021-22 cost BELOW the historical average.)")


print("\nTASK 6: Saving Models...")
# Save models to disk
for target in targets:
    # Create a safe filename
    safe_name = target.split(' (')[0].replace(' ', '_').replace('>', '').replace('/', '_').lower()
    model_path = os.path.join(models_dir, f"model_{safe_name}.pkl")
    joblib.dump(final_models[target], model_path)
    print(f"Saved: {model_path}")

print("\nAll tasks completed successfully!")
