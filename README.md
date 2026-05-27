# NYC Airbnb 2019 — Full-Stack Data Analysis Portfolio

End-to-end analysis of ~49,000 NYC Airbnb listings, spanning the full analytics workflow: **EDA → SQL → ML → interactive dashboard → BI reporting**.

This repository contains two iterations of the project:
- **Original coursework** (Aug – Dec 2023, Oklahoma State University): exploratory analysis with descriptive statistics and visualizations.
- **Extended portfolio version** (2026): a complete data-stack rebuild with predictive modeling, an interactive Streamlit app, SQL analytics, and BI-ready exports.

## Headline results

- **48,608 listings** analyzed across five boroughs after data cleaning.
- **Manhattan averages $179/night, 2.1× the Bronx** ($85). Brooklyn sits at $118.
- **Commercial operators are real:** the top 1% of hosts control ~10% of listings; the single largest operator (Sonder NYC) runs **327 properties**.
- **XGBoost price model** (hyperparameter-tuned via RandomizedSearchCV): **R² = 0.44, MAE = $45.40** on a held-out test set, beating Linear Regression (R² = 0.29) and Random Forest (R² = 0.43).
- **Room type and Manhattan-vs-not** account for ~90% of model importance — pricing is dominated by *what kind of space* and *which borough*.

## Project structure

```
.
├── README.md                        ← you are here
├── requirements.txt
│
├── AB_NYC_2019.csv                  ← source dataset
├── Airbnb Analysis.pdf              ← original 2023 course report
├── python pp final code .py         ← original 2023 script
│
├── 01_analysis.ipynb                ← EDA notebook with narration (start here)
├── 02_sql_analysis.sql              ← analytical SQL queries
├── 02_sql_demo.py                   ← runs the SQL queries against SQLite
├── 03_ml_pipeline.py                ← XGBoost training + tuning
├── 04_dashboard.py                  ← Streamlit interactive dashboard
│
├── POWERBI_GUIDE.md                 ← step-by-step Power BI build instructions
├── powerbi_borough_summary.csv      ← BI-ready dimension table
├── powerbi_neighborhood_summary.csv
├── powerbi_room_summary.csv
│
└── (generated at runtime)
    ├── airbnb_clean.csv             ← cleaned dataset (from notebook)
    ├── airbnb.db                    ← SQLite database (from SQL demo)
    ├── xgb_price_model.pkl          ← trained model (from ML pipeline)
    └── model_metadata.json          ← model metadata for dashboard
```

## How to run

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Run the EDA notebook (produces airbnb_clean.csv)
jupyter notebook 01_analysis.ipynb

# 2. Run the SQL analytics
python 02_sql_demo.py

# 3. Train the ML model (takes ~3 minutes)
python 03_ml_pipeline.py

# 4. Launch the interactive dashboard
streamlit run 04_dashboard.py
```

## What each piece demonstrates

| File | Skills demonstrated |
|------|---------------------|
| `01_analysis.ipynb` | EDA, pandas, data cleaning, matplotlib, seaborn, plotly, technical writing |
| `02_sql_analysis.sql` | SQL: CTEs, window functions (ROW_NUMBER, PERCENT_RANK), conditional aggregation, multi-step analytical queries |
| `02_sql_demo.py` | Python ↔ SQL integration, SQLite, sqlite3 |
| `03_ml_pipeline.py` | scikit-learn, XGBoost, RandomizedSearchCV, K-fold cross-validation, feature engineering, log-target transformation, model serialization |
| `04_dashboard.py` | Streamlit, interactive web apps, model deployment, plotly |
| `POWERBI_GUIDE.md` | BI design thinking, dashboard layout, dimensional modeling |

## Key findings

### 1. Geography drives pricing harder than anything else
Lat/lon coordinates outweigh any single review or availability metric for predicting price. The map view in the Streamlit dashboard makes this immediately visible: a dense, high-priced cluster in lower-to-midtown Manhattan, a high-priced ridge along the Brooklyn waterfront, with prices fading outward.

### 2. The market is segmented, not casual
A SQL segmentation of hosts into casual (1 listing), multi-host (2–4 listings), and commercial (5+ listings) reveals:

| Segment | Hosts | Listings | Share | Avg price | Avg avail |
|---------|-------|----------|-------|-----------|-----------|
| 1 listing (casual) | 32,122 | 32,122 | 66.1% | $146.64 | 78.5 days |
| 2-4 listings (multi-host) | 4,625 | 10,919 | 22.5% | $119.15 | 143.4 days |
| 5+ listings (commercial) | 504 | 5,567 | 11.5% | $154.11 | 247.7 days |

Commercial operators are **3× more available** than casual hosts (247 vs 78 days/year) — the difference between a hotel-like operation and someone renting out their spare room a few weeks a year.

### 3. Price is bounded by category, refined by location
XGBoost feature importance shows that room type alone explains ~63% of the model's predictive power, and Manhattan-or-not explains another ~27%. Numeric features (lat/lon, availability, reviews) refine within those buckets but don't override them.

## Methodology notes

- **Outliers and missingness.** Dropped 11 listings with `price == 0` (data errors) and capped the modeling range at $10–$1000 (raw max is $10,000/night, almost certainly typos). Filled `reviews_per_month` nulls with 0 (they correspond exactly to listings with no reviews — not random missingness).
- **Target transformation.** Used `log1p(price)` as the target for modeling because the raw distribution is heavily right-skewed. Predictions are transformed back to dollars with `expm1` for evaluation and display.
- **Hyperparameter tuning.** RandomizedSearchCV with 20 sampled configurations × 4-fold CV (80 total fits) on the XGBoost search space. Best config: 600 trees, max_depth=8, learning_rate=0.03, subsample=0.9.
- **Reproducibility.** Every script uses a fixed `random_state=42` for splits and model initialization.

## Dataset

NYC Airbnb Open Data (2019). Available on [Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data).

## Original (2023) coursework

`Airbnb Analysis.pdf` and `python pp final code .py` are preserved from the original Oklahoma State University course project. The 2026 extension adds the full data stack on top — see the project structure above.
