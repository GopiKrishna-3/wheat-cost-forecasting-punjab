# AI-Based Agricultural Input Cost Forecasting

This project forecasts agricultural input costs for wheat cultivation in Punjab, India, using historical government cost data and interpretable machine learning models.

Live app: https://wheat-cost-forecasting-punjab-wymkk52ahvsvjwg9tbe9fj.streamlit.app/

## Overview

The application helps farmers, agri-business stakeholders, and planners estimate future costs for:

- Seed cost
- Fertilizer cost
- Irrigation charges
- Human labour cost
- Total cost of cultivation (C2)

It uses a lightweight, transparent forecasting approach based on historical trends and linear regression models, with SHAP-based explainability to show how the time trend affects predictions.

## Project Objective

To provide a practical financial planning tool that offers baseline cost forecasts for wheat cultivation in Punjab using official CACP-based historical data.

## Dataset

- Source: CACP / Directorate of Economics & Statistics, Government of India
- Coverage: Punjab wheat crop
- Years: 2017-18 to 2021-22
- Metrics included:
  - Total Cost of Cultivation C2 (Rs./Hectare)
  - Fertilizer cost (Rs./Hectare)
  - Irrigation charges (Rs./Hectare)
  - Seed cost (Rs./Hectare)
  - Human labour cost (Rs./Hectare)

## Methodology

- Data cleaning and reshaping from raw output files
- Feature engineering with a time index
- Training separate linear regression models for each target variable
- Forecasting future years using trend-based extrapolation
- SHAP-based explainability to interpret the impact of the time trend

## Tech Stack

- Python
- Pandas
- NumPy
- scikit-learn
- Joblib
- SHAP
- Matplotlib
- Seaborn
- Streamlit

## Repository Structure

- `app.py` – Streamlit dashboard
- `process_data.py` – preprocessing and data preparation
- `train_forecast.py` – model training and evaluation
- `punjab_wheat_cleaned_wide.csv` – cleaned dataset
- `models/` – saved trained model files
- `forecast_results.csv` – forecast output
- `model_evaluation.csv` – model evaluation metrics

## Running the Project Locally

1. Clone the repository.
2. Navigate to the project directory.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser, typically at:

```text
http://localhost:8501
```

## Notes

This tool is designed as a planning aid rather than a guarantee of future agricultural costs. Forecasting is based only on historical trend signals and should be used together with domain expertise and current market information.

## Project Documentation

Additional project details and methodology notes are available in `PROJECT_DOCUMENTATION.md`.
