"""
NYC Airbnb Analysis (2019)
==========================
Exploratory data analysis and price prediction on ~49K NYC Airbnb listings.

Pipeline:
    1. Load and clean the data (handle nulls, drop invalid prices, cap outliers).
    2. Compute descriptive statistics by borough and room type.
    3. Generate visualizations: price comparisons, distributions, correlation
       heatmap, and an interactive geospatial map.
    4. Train regression models (Linear Regression, Random Forest) to predict
       nightly price, and report feature importance.

Usage:
    python airbnb_analysis.py

Expects AB_NYC_2019.csv in the same directory as this script.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "AB_NYC_2019.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Cap extreme prices for modeling and most visualizations.
# Max in raw data is $10,000/night — clearly outliers that distort everything.
PRICE_FLOOR = 10
PRICE_CEILING = 1000

sns.set_theme(style="whitegrid", palette="muted")


# ---------------------------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------------------------
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw Airbnb CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Download AB_NYC_2019.csv and place it next to this script."
        )
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean nulls, drop invalid rows, and filter outliers for modeling."""
    df = df.copy()

    # reviews_per_month is null exactly when there are no reviews — fill with 0.
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

    # Drop listings with price <= 0 (data errors) and cap extreme outliers.
    n_before = len(df)
    df = df[(df["price"] >= PRICE_FLOOR) & (df["price"] <= PRICE_CEILING)]
    n_dropped = n_before - len(df)
    print(f"Cleaning: dropped {n_dropped} rows outside ${PRICE_FLOOR}-${PRICE_CEILING}")

    # Drop the few rows with no name/host_name — they're noise.
    df = df.dropna(subset=["name", "host_name"])

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Descriptive analysis
# ---------------------------------------------------------------------------
def summarize_by_borough(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-borough summary: count, mean/median price, avg availability."""
    summary = (
        df.groupby("neighbourhood_group")
        .agg(
            listings=("id", "count"),
            mean_price=("price", "mean"),
            median_price=("price", "median"),
            avg_availability=("availability_365", "mean"),
            avg_reviews=("number_of_reviews", "mean"),
        )
        .round(1)
        .sort_values("mean_price", ascending=False)
    )
    return summary


def host_concentration(df: pd.DataFrame) -> dict:
    """How concentrated is the market? Top-N hosts as share of all listings."""
    host_counts = df["host_id"].value_counts()
    total = len(df)
    return {
        "total_hosts": len(host_counts),
        "multi_listing_hosts": int((host_counts > 1).sum()),
        "top_1pct_share": float(
            host_counts.head(max(1, len(host_counts) // 100)).sum() / total
        ),
        "largest_host_listings": int(host_counts.iloc[0]),
    }


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def plot_price_by_borough(df: pd.DataFrame, save: bool = True) -> None:
    """Side-by-side: mean price by borough, and price distribution by borough."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    mean_price = (
        df.groupby("neighbourhood_group")["price"].mean().sort_values(ascending=False)
    )
    sns.barplot(
        x=mean_price.index,
        y=mean_price.values,
        hue=mean_price.index,
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Mean Nightly Price by Borough")
    axes[0].set_ylabel("Mean price (USD)")
    axes[0].set_xlabel("")
    for i, v in enumerate(mean_price.values):
        axes[0].text(i, v + 2, f"${v:.0f}", ha="center", fontweight="bold")

    sns.boxplot(
        data=df,
        x="neighbourhood_group",
        y="price",
        hue="neighbourhood_group",
        legend=False,
        ax=axes[1],
        order=mean_price.index,
    )
    axes[1].set_title("Price Distribution by Borough")
    axes[1].set_ylabel("Price (USD)")
    axes[1].set_xlabel("")

    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_DIR / "01_price_by_borough.png", dpi=120)
    plt.close()


def plot_price_by_borough_and_room(df: pd.DataFrame, save: bool = True) -> None:
    """Mean price split by borough and room type — shows the room-type premium."""
    grouped = (
        df.groupby(["neighbourhood_group", "room_type"])["price"]
        .mean()
        .unstack()
        .round(0)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    grouped.plot(kind="bar", ax=ax, edgecolor="white")
    ax.set_title("Mean Price by Borough and Room Type")
    ax.set_ylabel("Mean price (USD)")
    ax.set_xlabel("")
    ax.legend(title="Room type")
    plt.xticks(rotation=0)
    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_DIR / "02_price_by_borough_and_room.png", dpi=120)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True) -> None:
    """Correlation between numeric features — which actually drive price?"""
    numeric_cols = [
        "price",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "latitude",
        "longitude",
    ]
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_DIR / "03_correlation_heatmap.png", dpi=120)
    plt.close()


def plot_price_distribution(df: pd.DataFrame, save: bool = True) -> None:
    """Log-scale price distribution — raw is too skewed to be readable."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df["price"], bins=50, ax=axes[0], color="steelblue")
    axes[0].set_title("Price Distribution (linear scale)")
    axes[0].set_xlabel("Price (USD)")

    sns.histplot(
        np.log10(df["price"]), bins=50, ax=axes[1], color="steelblue"
    )
    axes[1].set_title("Price Distribution (log10 scale)")
    axes[1].set_xlabel("log10(Price)")

    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_DIR / "04_price_distribution.png", dpi=120)
    plt.close()


def plot_geospatial_map(df: pd.DataFrame, save: bool = True) -> None:
    """Interactive map: listings colored by price (log-scaled for visibility)."""
    sample = df.sample(min(10000, len(df)), random_state=42)
    fig = px.scatter_mapbox(
        sample,
        lat="latitude",
        lon="longitude",
        color=np.log10(sample["price"]),
        size_max=8,
        zoom=10,
        center={"lat": 40.73, "lon": -73.95},
        mapbox_style="open-street-map",
        color_continuous_scale="Viridis",
        hover_data={"price": True, "room_type": True, "neighbourhood": True},
        title="NYC Airbnb Listings — Price (log scale)",
        height=700,
    )
    fig.update_layout(coloraxis_colorbar=dict(title="log10(price)"))
    if save:
        fig.write_html(OUTPUT_DIR / "05_geospatial_map.html")


# ---------------------------------------------------------------------------
# Predictive modeling
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame):
    """One-hot encode categoricals, return X (DataFrame) and y (Series)."""
    feature_cols_numeric = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "latitude",
        "longitude",
    ]
    feature_cols_categorical = ["neighbourhood_group", "room_type"]

    encoder = OneHotEncoder(sparse_output=False, drop="first")
    cat_encoded = encoder.fit_transform(df[feature_cols_categorical])
    cat_df = pd.DataFrame(
        cat_encoded,
        columns=encoder.get_feature_names_out(feature_cols_categorical),
        index=df.index,
    )

    X = pd.concat([df[feature_cols_numeric].reset_index(drop=True),
                   cat_df.reset_index(drop=True)], axis=1)
    y = df["price"].reset_index(drop=True)
    return X, y


def train_and_evaluate(df: pd.DataFrame) -> dict:
    """Train Linear Regression + Random Forest, report metrics and importance."""
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    # Linear Regression baseline
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    results["linear_regression"] = {
        "r2": r2_score(y_test, lr_pred),
        "mae": mean_absolute_error(y_test, lr_pred),
    }

    # Random Forest — captures non-linearities
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=20, n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results["random_forest"] = {
        "r2": r2_score(y_test, rf_pred),
        "mae": mean_absolute_error(y_test, rf_pred),
    }

    # Feature importance from the Random Forest
    importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    results["top_features"] = importance.head(10).round(4).to_dict()

    # Save feature-importance plot
    plt.figure(figsize=(10, 6))
    importance.head(10).sort_values().plot(kind="barh", color="teal")
    plt.title("Top 10 Features Predicting Price (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_feature_importance.png", dpi=120)
    plt.close()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("NYC AIRBNB ANALYSIS (2019)")
    print("=" * 70)

    df_raw = load_data()
    print(f"\nLoaded {len(df_raw):,} raw listings, {df_raw.shape[1]} columns")

    df = clean_data(df_raw)
    print(f"After cleaning: {len(df):,} listings\n")

    # Descriptive
    print("-" * 70)
    print("BOROUGH SUMMARY")
    print("-" * 70)
    summary = summarize_by_borough(df)
    print(summary.to_string())

    print("\n" + "-" * 70)
    print("HOST CONCENTRATION")
    print("-" * 70)
    hc = host_concentration(df)
    print(f"  Total unique hosts:        {hc['total_hosts']:,}")
    print(f"  Hosts with >1 listing:     {hc['multi_listing_hosts']:,} "
          f"({hc['multi_listing_hosts']/hc['total_hosts']:.1%})")
    print(f"  Top 1% of hosts control:   {hc['top_1pct_share']:.1%} of listings")
    print(f"  Largest single host has:   {hc['largest_host_listings']} listings")

    # Visualizations
    print("\n" + "-" * 70)
    print("GENERATING VISUALIZATIONS")
    print("-" * 70)
    plot_price_by_borough(df)
    print("  ✓ 01_price_by_borough.png")
    plot_price_by_borough_and_room(df)
    print("  ✓ 02_price_by_borough_and_room.png")
    plot_correlation_heatmap(df)
    print("  ✓ 03_correlation_heatmap.png")
    plot_price_distribution(df)
    print("  ✓ 04_price_distribution.png")
    plot_geospatial_map(df)
    print("  ✓ 05_geospatial_map.html (interactive)")

    # Predictive modeling
    print("\n" + "-" * 70)
    print("PRICE PREDICTION MODELS")
    print("-" * 70)
    results = train_and_evaluate(df)
    print(f"  Linear Regression:  R² = {results['linear_regression']['r2']:.3f}, "
          f"MAE = ${results['linear_regression']['mae']:.2f}")
    print(f"  Random Forest:      R² = {results['random_forest']['r2']:.3f}, "
          f"MAE = ${results['random_forest']['mae']:.2f}")
    print("  ✓ 06_feature_importance.png")

    print("\n  Top 5 features driving price:")
    for i, (feat, imp) in enumerate(list(results["top_features"].items())[:5], 1):
        print(f"    {i}. {feat:40s} {imp:.3f}")

    print("\n" + "=" * 70)
    print(f"Done. All outputs in: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
