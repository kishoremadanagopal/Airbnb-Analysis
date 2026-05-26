# NYC Airbnb Analysis

Analysis of ~49,000 New York City Airbnb listings (2019 open data).

This repository contains two iterations of the project:

- **Original coursework** (Aug – Dec 2023, Oklahoma State University): exploratory data analysis with descriptive statistics and visualizations, delivered as a Python script and PDF report.
- **Extended analysis** (2026): refactored code, expanded EDA, interactive geospatial visualization, and a price prediction model built on top of the original work.

## Repository structure

```
.
├── AB_NYC_2019.csv                    # Source dataset
│
├── Airbnb Analysis.pdf                # Original course report
├── python pp final code .py           # Original analysis script
│
├── v2_extended_analysis/              # Post-course extension
│   ├── airbnb_analysis.py             # Refactored modular pipeline
│   ├── README.md                      # Extension-specific details
│   ├── requirements.txt
│   └── outputs/                       # Generated charts + interactive map
│       ├── 01_price_by_borough.png
│       ├── 02_price_by_borough_and_room.png
│       ├── 03_correlation_heatmap.png
│       ├── 04_price_distribution.png
│       ├── 05_geospatial_map.html
│       └── 06_feature_importance.png
│
└── README.md                          # This file
```

## Original coursework (2023)

The original project performed exploratory data analysis on the NYC Airbnb dataset using pandas, NumPy, Matplotlib, Seaborn, and Plotly. It examined the relationship between price and minimum nights, the interaction between availability and pricing, and the relationship between price and review count across NYC's five boroughs.

See `Airbnb Analysis.pdf` for the original report and `python pp final code .py` for the original analysis script.

## Extended analysis (2026)

Returning to the project after the course, I identified several areas to extend the original work:

**Code quality.** Refactored the script into modular functions with proper data cleaning (handling nulls in `reviews_per_month`, dropping invalid `price == 0` rows, capping extreme outliers), relative file paths, and dependency management via `requirements.txt`.

**Deeper EDA.** Added a correlation matrix across all numeric features, a log-scale price distribution (the raw distribution is too skewed to read), and host-concentration metrics to test whether the NYC short-term rental market is dominated by commercial operators.

**Geospatial visualization.** The original analysis never used the latitude/longitude columns. The extension adds an interactive Plotly map showing all listings colored by price, which makes neighborhood-level patterns immediately visible.

**Predictive modeling.** Added Linear Regression and Random Forest models predicting nightly price from location, room type, availability, and review activity, with reported R², MAE, and feature importance.

### Key findings from the extended analysis

- **48,608 listings** after cleaning (287 invalid rows removed).
- **Manhattan averages $179/night, 2.1× the Bronx** ($85/night).
- **13.8% of hosts manage multiple listings**; the **top 1% control 10.1%** of all listings; the largest single host operates **327 listings** — evidence of meaningful commercial activity in the market.
- **Random Forest** predicts price with **R² = 0.43, MAE ≈ $50**, outperforming a Linear Regression baseline (R² = 0.31).
- **Location and room type dominate**, together accounting for over 60% of model feature importance.

See `v2_extended_analysis/README.md` for setup and run instructions.

## Dataset

NYC Airbnb Open Data (2019), available on [Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data).
