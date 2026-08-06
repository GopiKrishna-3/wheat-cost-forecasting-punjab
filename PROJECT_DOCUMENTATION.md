# AI-Based Agricultural Input Cost Forecasting

## 1. PROJECT OVERVIEW
**Title:** AI-Based Agricultural Input Cost Forecasting
**Objective:** To forecast agricultural input costs (Seed, Fertilizer, Irrigation, Human Labour, and Total Cost of Cultivation) to aid financial planning for farmers and agri-businesses.
**Scope:** Wheat crop in the state of Punjab, India.

## 2. DATASET
- **Source:** CACP (Commission for Agricultural Costs & Prices) / Directorate of Economics & Statistics, Ministry of Agriculture, Government of India.
- **Coverage:** Punjab, Wheat, spanning 5 years (2017-18 to 2021-22).
- **Acquisition:** The data was manually compiled from official government yearly reports. 
- **Key Metrics Extracted:** 
  - Total Cost of Cultivation C2 (Rs./Hectare)
  - Fertilizer cost (Rs./Hectare)
  - Irrigation charges (Rs./Hectare)
  - Seed cost (Rs./Hectare)
  - Human Labour cost (Rs./Hectare)
- **Data Quality & Distinctiveness:** This dataset is distinct from generic public datasets as it required manual extraction and disambiguation from raw, multi-sheet government Excel/PDF reports. For example, during data cleaning, a column collision bug between "quantities" (e.g., kg/hectare) and "rates" (Rs./unit) was identified and resolved to ensure only the true cost components were isolated for modeling.

## 3. METHODOLOGY
- **Data Cleaning:** The raw dataset was pivoted from a long/tidy format into a wide format. Numeric type checks were enforced and no missing values were present in the final target metrics.
- **EDA (Exploratory Data Analysis):** Trend line charts and year-over-year percentage change analyses were conducted to understand historical cost behaviors.
- **Feature Engineering:** A `time_index` (values 0 to 4, representing the 5 years) was created as the primary feature to represent time mathematically.
- **Modeling:** 5 separate Linear Regression models were trained (one for each target variable).
- **Model Selection Justification:** Linear Regression was explicitly chosen because there are only 5 data points available. Simpler, parametric models are highly appropriate and interpretable for small sample sizes, whereas complex models (like Random Forests or Gradient Boosting) would heavily overfit the data.
- **Evaluation:** Leave-One-Out Cross-Validation (LOOCV) was used as the primary evaluation metric. Given the extremely small sample size, a standard train/test split is not statistically meaningful on its own. A supplementary train/test split (Train: 2017-18 to 2020-21, Test: 2021-22) was also calculated for comparison.
- **Explainability:** SHAP (SHapley Additive exPlanations) using `LinearExplainer` was applied to quantify exactly how much the time trend pushes the predictions up or down relative to the 5-year average.

## 4. RESULTS

### Model Evaluation (LOOCV & Train/Test Split)
*Note: In LOOCV, the Mean Absolute Error (MAE) mathematically equals the Root Mean Squared Error (RMSE) because the error of a single left-out point squared, then square-rooted, equals its absolute value.*

| Target | LOOCV MAE / RMSE | Train/Test MAE / RMSE |
| :--- | :--- | :--- |
| **Total Cost of Cultivation C2** | 3,971.09 | 6,069.91 |
| **Fertilizer cost** | 421.13 | 354.30 |
| **Irrigation charges** | 34.95 | 27.52 |
| **Seed cost** | 65.96 | 144.27 |
| **Human Labour cost** | 417.10 | 334.29 |

### Forecasted Values
| Year | Total Cost C2 | Fertilizer cost | Irrigation charges | Seed cost | Human Labour cost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2022-23** | ₹72,747.07 | ₹5,829.88 | ₹781.71 | ₹3,281.03 | ₹5,834.11 |
| **2023-24** | ₹73,613.75 | ₹5,869.31 | ₹810.86 | ₹3,426.10 | ₹5,867.07 |

### SHAP Explainability Findings
Based on the Linear Regression coefficients (which SHAP explains linearly):
- **Total Cost (C2)** is trending **upward** by approximately **₹866.68 per year**. 
- The time trend consistently pushes all sub-components up, acting as an isolated baseline driver for input inflation.

## 5. KEY FINDINGS
- **Overall Trends:** The Total Cost of Cultivation (C2) shows a steady upward trend from 2017-18 to 2020-21 (peaking at ₹72,250.53). 
- **Anomaly in 2021-22:** There is a notable dip in the Total Cost in 2021-22, representing a **-3.9% Year-over-Year decline** (down to ₹69,452.42). This anomaly warrants further economic investigation.
- **Component Trends:** Despite the overall dip in 2021-22, sub-components like Fertilizer, Seed, and Human Labour costs continue to exhibit consistent historical upward trends across the 5 years.
- **Practical Implications:** For farmers and agri-businesses, this suggests that core operational inputs are persistently inflating. Budgets must account for guaranteed rises in material and labor inputs regardless of the final Total Cost output.

## 6. LIMITATIONS
- **Data Scarcity:** The fundamental constraint of this model is the reliance on only 5 years of historical data, which limits deep statistical reliability.
- **Linear Extrapolation:** Linear Regression assumes a constant rate of change. This assumption often fails over longer time horizons. Extrapolating beyond 2-3 years (e.g., forecasting for 2027) is not recommended without updated data.
- **Omitted Variable Bias:** The current model does not account for external macroeconomic factors such as weather anomalies, shifts in government policy, global commodity price shocks, or subsidy changes, all of which heavily influence real-world agricultural costs.
- **Single-Feature Model:** By only utilizing time (`time_index`) as a feature, the model ignores farm-specific variations such as soil type, irrigation methodology, or farm size.

## 7. CONCLUSION
This project successfully built a functioning, data-driven forecasting tool to project the costs of critical agricultural inputs for Wheat farming in Punjab. By leveraging an interactive Streamlit dashboard ("Agricultural Ledger") and SHAP explainability, the tool provides transparent baseline financial estimates based purely on recent historical momentum. 

However, users must be aware that this tool is designed strictly as a **financial planning aid**, not a guarantee. It should be used in conjunction with domain expertise and up-to-date market intelligence.

## 8. TECHNICAL DELIVERABLES
- **GitHub Repository:** [wheat-cost-forecasting-punjab](https://github.com/GopiKrishna-3/wheat-cost-forecasting-punjab)
- **Live Application (Deployed):** [wheat-cost-forecasting-punjab](https://wheat-cost-forecasting-punjab-v72kbjayiiga4uu7cbergq.streamlit.app)
- **Trained Models:** 5 saved `.pkl` files (e.g., `model_total_cost_of_cultivation_c2.pkl`)
- **Key Scripts:**
  - `process_data.py`: Handles ETL and reshaping the raw CACP data.
  - `train_forecast.py`: Generates the LOOCV evaluations and trains the final prediction models.
  - `app.py`: The interactive frontend web application.
