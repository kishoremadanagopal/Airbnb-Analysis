# NYC Airbnb Analysis (2019)

End-to-end analysis of ~49,000 New York City Airbnb listings: exploratory data analysis, geospatial visualization, and a price prediction model.

## Overview

This project analyzes the public NYC Airbnb 2019 dataset to answer three questions:
1. **How do prices and availability vary across boroughs and room types?**
2. **Is the NYC short-term rental market dominated by commercial operators?**
3. **Can we predict the nightly price of a listing from its features?**

## Key findings

- **48,608 listings** analyzed across five boroughs (after cleaning 287 invalid rows).
- **Manhattan is the most expensive borough** at a mean of $179/night, **2.1x** the Bronx ($85/night).
- **13.8% of hosts manage multiple listings**, and the **top 1% of hosts control 10.1%** of all listings — evidence that commercial operators play a meaningful role in the NYC market. The single largest host operates **327 listings**.
- **Random Forest model** predicts nightly price with **R² = 0.43** and **MAE ≈ $50**, outperforming a linear baseline (R² = 0.31). The strongest predictors are **location (latitude/longitude)** and **room type** — together accounting for over 60% of model importance.

## Project structure

```
.
├── airbnb_analysis.py    # Main pipeline (cleaning → EDA → modeling)
├── AB_NYC_2019.csv       # Source data (download separately, see below)
├── requirements.txt
├── README.md
└── outputs/              # Generated charts and the interactive map
    ├── 01_price_by_borough.png
    ├── 02_price_by_borough_and_room.png
    ├── 03_correlation_heatmap.png
    ├── 04_price_distribution.png
    ├── 05_geospatial_map.html
    └── 06_feature_importance.png
```

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the dataset (Kaggle: NYC Airbnb Open Data) and place it next to the script
#    https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data

# 3. Run the pipeline
python airbnb_analysis.py
```

Outputs are written to `outputs/`. The geospatial map is an interactive HTML file — open it in any browser to pan, zoom, and hover over individual listings.

## Methodology notes

- **Outlier handling.** Raw prices range from $0 to $10,000/night. The pipeline drops listings priced at $0 (data errors, n=11) and caps the range at $10–$1000 for modeling, which removes ~0.5% of the data and prevents extreme values from dominating both visualizations and model loss.
- **Missing data.** `reviews_per_month` is null exactly when a listing has zero reviews, so it's filled with 0 rather than dropped (preserves ~20% of the dataset).
- **Model evaluation.** 80/20 train/test split with a fixed random seed for reproducibility. Random Forest uses 100 trees, max depth 20. R² and MAE reported on the held-out test set.

## What's next

- Add temporal features by joining the host-level review timestamps to capture seasonality.
- Try gradient boosting (XGBoost / LightGBM) — typically lifts R² by 5–10 points on tabular data of this shape.
- Build a Streamlit dashboard wrapping the model so users can input listing features and see a predicted price.
