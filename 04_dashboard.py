"""
NYC Airbnb 2019 — Interactive Streamlit Dashboard
==================================================
Run with:
    streamlit run 04_dashboard.py

Features:
    - Sidebar filters: borough, room type, price range
    - Live KPI cards: count, mean price, median price, avg availability
    - Interactive charts: price by borough, room-type mix, geospatial map
    - Live price predictor backed by the tuned XGBoost model from 03_ml_pipeline.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

HERE = Path(__file__).parent
CSV_PATH = HERE / "airbnb_clean.csv"
MODEL_PATH = HERE / "xgb_price_model.pkl"
METADATA_PATH = HERE / "model_metadata.json"

st.set_page_config(
    page_title="NYC Airbnb Analytics", page_icon="🏙️", layout="wide"
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None, None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return model, metadata


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🏙️ NYC Airbnb 2019 — Analytics Dashboard")
st.caption(
    "Interactive exploration of ~49K NYC Airbnb listings, with a live XGBoost "
    "price predictor. Use the sidebar to filter the data."
)

df = load_data()
model, metadata = load_model()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")
boroughs = sorted(df["neighbourhood_group"].unique())
selected_boroughs = st.sidebar.multiselect(
    "Borough", boroughs, default=boroughs
)

room_types = sorted(df["room_type"].unique())
selected_rooms = st.sidebar.multiselect(
    "Room type", room_types, default=room_types
)

price_min, price_max = int(df["price"].min()), int(df["price"].max())
price_range = st.sidebar.slider(
    "Price range (USD/night)",
    min_value=price_min, max_value=price_max,
    value=(price_min, min(500, price_max)),
)

# Apply filters
filtered = df[
    df["neighbourhood_group"].isin(selected_boroughs)
    & df["room_type"].isin(selected_rooms)
    & df["price"].between(*price_range)
]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Filtered listings:** {len(filtered):,}")

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
if len(filtered) == 0:
    st.warning("No listings match the current filters. Adjust the sidebar.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Listings", f"{len(filtered):,}")
c2.metric("Mean price", f"${filtered['price'].mean():.0f}/night")
c3.metric("Median price", f"${filtered['price'].median():.0f}/night")
c4.metric("Avg availability", f"{filtered['availability_365'].mean():.0f} days")

st.markdown("---")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Price by Borough", "🏠 Room Type Mix", "🗺️ Map", "🤖 Price Predictor"]
)

# --- Tab 1: Price by Borough ---
with tab1:
    col1, col2 = st.columns(2)

    mean_price = (
        filtered.groupby("neighbourhood_group")["price"]
        .mean().sort_values(ascending=False).reset_index()
    )
    fig1 = px.bar(
        mean_price, x="neighbourhood_group", y="price",
        color="neighbourhood_group",
        labels={"neighbourhood_group": "Borough", "price": "Mean price (USD)"},
        title="Mean Nightly Price by Borough",
    )
    fig1.update_layout(showlegend=False)
    col1.plotly_chart(fig1, use_container_width=True)

    fig2 = px.box(
        filtered, x="neighbourhood_group", y="price",
        color="neighbourhood_group",
        labels={"neighbourhood_group": "Borough", "price": "Price (USD)"},
        title="Price Distribution by Borough",
    )
    fig2.update_layout(showlegend=False)
    col2.plotly_chart(fig2, use_container_width=True)

# --- Tab 2: Room Type Mix ---
with tab2:
    col1, col2 = st.columns(2)

    room_counts = (
        filtered.groupby(["neighbourhood_group", "room_type"])
        .size().reset_index(name="count")
    )
    fig3 = px.bar(
        room_counts, x="neighbourhood_group", y="count", color="room_type",
        labels={"neighbourhood_group": "Borough", "count": "Listings"},
        title="Listings by Borough × Room Type",
    )
    col1.plotly_chart(fig3, use_container_width=True)

    room_price = (
        filtered.groupby("room_type")["price"]
        .mean().reset_index().sort_values("price", ascending=False)
    )
    fig4 = px.bar(
        room_price, x="room_type", y="price", color="room_type",
        labels={"room_type": "Room type", "price": "Mean price (USD)"},
        title="Mean Price by Room Type",
    )
    fig4.update_layout(showlegend=False)
    col2.plotly_chart(fig4, use_container_width=True)

# --- Tab 3: Map ---
with tab3:
    sample_size = min(5000, len(filtered))
    sample = filtered.sample(sample_size, random_state=42)
    st.caption(
        f"Showing a random sample of {sample_size:,} listings (of "
        f"{len(filtered):,} filtered) for performance."
    )
    fig_map = px.scatter_mapbox(
        sample, lat="latitude", lon="longitude",
        color=np.log10(sample["price"]),
        zoom=10, center={"lat": 40.73, "lon": -73.95},
        mapbox_style="open-street-map", color_continuous_scale="Viridis",
        hover_data={"price": True, "room_type": True, "neighbourhood": True},
        height=600,
    )
    fig_map.update_layout(coloraxis_colorbar=dict(title="log10(price)"))
    st.plotly_chart(fig_map, use_container_width=True)

# --- Tab 4: Price Predictor ---
with tab4:
    st.subheader("🤖 Predict the nightly price of a hypothetical listing")
    st.caption(
        "Powered by the tuned XGBoost model (trained in `03_ml_pipeline.py`). "
        "Adjust the inputs to see the predicted nightly price."
    )

    if model is None:
        st.error(
            "Model files not found. Run `python 03_ml_pipeline.py` first to "
            "train and save the model."
        )
    else:
        test_metrics = metadata["test_metrics"]
        st.info(
            f"**Model performance on held-out test set:**  "
            f"R² = {test_metrics['r2']:.3f}  |  "
            f"MAE = ${test_metrics['mae']:.2f}  |  "
            f"RMSE = ${test_metrics['rmse']:.2f}"
        )

        col1, col2 = st.columns(2)

        with col1:
            borough = st.selectbox("Borough", metadata["boroughs"])
            room_type = st.selectbox("Room type", metadata["room_types"])
            minimum_nights = st.number_input(
                "Minimum nights", min_value=1, max_value=365, value=2
            )
            availability_365 = st.slider(
                "Availability (days/year)", 0, 365, 180
            )

        with col2:
            number_of_reviews = st.number_input(
                "Number of reviews", min_value=0, max_value=1000, value=10
            )
            reviews_per_month = st.number_input(
                "Reviews per month",
                min_value=0.0, max_value=20.0, value=0.5, step=0.1
            )
            host_listings = st.number_input(
                "Listings managed by host", min_value=1, max_value=500, value=1
            )

            # Use Manhattan centroid as a sensible default
            lat = st.number_input(
                "Latitude", min_value=40.5, max_value=41.0,
                value=40.728, step=0.001, format="%.3f"
            )
            lon = st.number_input(
                "Longitude", min_value=-74.3, max_value=-73.7,
                value=-73.99, step=0.001, format="%.3f"
            )

        # Build a feature row matching the training schema exactly.
        row = {col: 0 for col in metadata["feature_columns"]}
        row["latitude"] = lat
        row["longitude"] = lon
        row["minimum_nights"] = minimum_nights
        row["number_of_reviews"] = number_of_reviews
        row["reviews_per_month"] = reviews_per_month
        row["calculated_host_listings_count"] = host_listings
        row["availability_365"] = availability_365
        # Median neighborhood count as a default frequency-encoding value
        nbhd_counts = metadata["nbhd_counts"]
        row["nbhd_listings_count"] = int(np.median(list(nbhd_counts.values())))
        row[f"borough_{borough}"] = 1
        row[f"room_{room_type}"] = 1

        X_pred = pd.DataFrame([row])[metadata["feature_columns"]]
        log_pred = float(model.predict(X_pred)[0])
        price_pred = float(np.expm1(log_pred))

        st.markdown("### Predicted nightly price")
        st.markdown(
            f"<h1 style='color:#1f77b4;text-align:center'>"
            f"${price_pred:,.2f}</h1>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Expected error: ±${test_metrics['mae']:.0f} on average "
            f"(test-set MAE)."
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Data: NYC Airbnb Open Data (2019, Kaggle). Models: scikit-learn + XGBoost. "
    "Dashboard: Streamlit + Plotly."
)
