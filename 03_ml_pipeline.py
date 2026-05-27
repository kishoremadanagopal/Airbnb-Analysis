"""
NYC Airbnb — Advanced Price Prediction Pipeline
================================================
Builds and compares multiple regression models to predict nightly price.

Pipeline:
    1. Load cleaned data (airbnb_clean.csv from notebook 01).
    2. Feature engineering: one-hot encode categoricals, log-transform target,
       add neighborhood-level frequency encoding.
    3. Train three models: Linear Regression (baseline), Random Forest,
       XGBoost with RandomizedSearchCV hyperparameter tuning.
    4. Evaluate on held-out test set with R², MAE, and RMSE.
    5. Save the best model + feature names for the Streamlit dashboard.

Usage:
    python 03_ml_pipeline.py
"""

import json
import pickle
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

HERE = Path(__file__).parent
CSV_PATH = HERE / "airbnb_clean.csv"
MODEL_PATH = HERE / "xgb_price_model.pkl"
METADATA_PATH = HERE / "model_metadata.json"

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Build the modeling matrix.

    Returns
    -------
    X : DataFrame of features
    y : Series of log-transformed price
    metadata : dict capturing encoders/categories so the dashboard can
               reproduce the same feature space at inference time.
    """
    df = df.copy()

    # Frequency encoding for neighbourhood — captures local market size.
    nbhd_counts = df["neighbourhood"].value_counts()
    df["nbhd_listings_count"] = df["neighbourhood"].map(nbhd_counts)

    numeric_features = [
        "latitude",
        "longitude",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "nbhd_listings_count",
    ]

    # One-hot encode borough and room type
    borough_dummies = pd.get_dummies(df["neighbourhood_group"], prefix="borough")
    room_dummies = pd.get_dummies(df["room_type"], prefix="room")

    X = pd.concat(
        [df[numeric_features].reset_index(drop=True),
         borough_dummies.reset_index(drop=True),
         room_dummies.reset_index(drop=True)],
        axis=1,
    )

    # Log-transform the target — distribution is heavily right-skewed.
    y = np.log1p(df["price"].reset_index(drop=True))

    metadata = {
        "numeric_features": numeric_features,
        "boroughs": sorted(df["neighbourhood_group"].unique().tolist()),
        "room_types": sorted(df["room_type"].unique().tolist()),
        "feature_columns": X.columns.tolist(),
        "nbhd_counts": nbhd_counts.to_dict(),
        "target_transform": "log1p",
    }
    return X, y, metadata


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(name: str, model, X_test, y_test_log) -> dict:
    """Evaluate on the original (untransformed) price scale."""
    y_pred_log = model.predict(X_test)
    y_test = np.expm1(y_test_log)
    y_pred = np.expm1(y_pred_log)

    return {
        "model": name,
        "r2": round(r2_score(y_test, y_pred), 4),
        "mae": round(mean_absolute_error(y_test, y_pred), 2),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
    }


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------
def tune_xgboost(X_train, y_train) -> XGBRegressor:
    """Randomized search over a sensible XGBoost hyperparameter grid."""
    param_dist = {
        "n_estimators": [200, 400, 600, 800],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0],
    }

    base = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        tree_method="hist",
    )

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=20,
        scoring="r2",
        cv=KFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print("NYC AIRBNB — ADVANCED ML PIPELINE")
    print("=" * 78)

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"{CSV_PATH.name} not found. Run notebook 01_analysis.ipynb first."
        )
    df = pd.read_csv(CSV_PATH)
    print(f"\nLoaded {len(df):,} rows.")

    X, y, metadata = engineer_features(df)
    print(f"Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} columns.")
    print(f"Target: log1p(price), mean={y.mean():.2f}, std={y.std():.2f}.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    results = []

    # ---- 1. Linear Regression baseline ----
    print("\n[1/3] Training Linear Regression (baseline)...")
    t0 = perf_counter()
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    results.append({**evaluate("Linear Regression", lr, X_test, y_test),
                    "train_seconds": round(perf_counter() - t0, 2)})

    # ---- 2. Random Forest ----
    print("[2/3] Training Random Forest...")
    t0 = perf_counter()
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_leaf=2,
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    results.append({**evaluate("Random Forest", rf, X_test, y_test),
                    "train_seconds": round(perf_counter() - t0, 2)})

    # ---- 3. XGBoost with hyperparameter tuning ----
    print("[3/3] Tuning XGBoost (RandomizedSearchCV, 20 iterations × 4-fold CV)...")
    t0 = perf_counter()
    search = tune_xgboost(X_train, y_train)
    xgb_best = search.best_estimator_
    results.append({**evaluate("XGBoost (tuned)", xgb_best, X_test, y_test),
                    "train_seconds": round(perf_counter() - t0, 2)})
    print(f"   Best CV R² (log-target): {search.best_score_:.4f}")
    print(f"   Best params: {search.best_params_}")

    # ---- Results table ----
    print("\n" + "=" * 78)
    print("MODEL COMPARISON (held-out test set, original price scale)")
    print("=" * 78)
    results_df = pd.DataFrame(results)[
        ["model", "r2", "mae", "rmse", "train_seconds"]
    ]
    print(results_df.to_string(index=False))

    # ---- Feature importance from XGBoost ----
    importance = pd.Series(
        xgb_best.feature_importances_, index=X.columns
    ).sort_values(ascending=False)
    print("\nTop 10 features (XGBoost gain importance):")
    for i, (feat, imp) in enumerate(importance.head(10).items(), 1):
        print(f"  {i:2d}. {feat:38s} {imp:.4f}")

    # ---- Save model and metadata for the dashboard ----
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(xgb_best, f)
    metadata["best_params"] = search.best_params_
    metadata["test_metrics"] = next(
        r for r in results if r["model"] == "XGBoost (tuned)"
    )
    metadata["all_results"] = results
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nSaved model       → {MODEL_PATH.name}")
    print(f"Saved metadata    → {METADATA_PATH.name}")
    print("=" * 78)


if __name__ == "__main__":
    main()
