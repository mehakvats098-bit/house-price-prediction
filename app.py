import streamlit as st
import pandas as pd
import joblib


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("data/house_prices.csv")


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("model/house_price_model.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🏠 House Price AI")

    st.caption(
        "Machine Learning Property Valuation"
    )

    st.divider()

    st.markdown("### 📌 Project")

    st.write(
        "Predict house prices using property "
        "and location features."
    )

    st.divider()

    st.markdown("### 🤖 Model")

    st.write("**Algorithm:** Linear Regression")
    st.write("**Task:** Regression")
    st.write("**Target:** Price (Lakh)")

    st.divider()

    st.markdown("### 📊 Dataset")

    st.write(f"**Properties:** {len(df)}")
    st.write(f"**Features:** {len(df.columns) - 2}")

    st.divider()

    st.caption("Built by Mehak • AIML")


# ==========================================
# PAGE TITLE
# ==========================================

st.title("🏠 House Price Prediction")

st.caption(
    "Machine learning powered property valuation dashboard"
)

st.divider()


# ==========================================
# MARKET OVERVIEW
# ==========================================

st.subheader("📊 Market Overview")

avg_price = df["Price_Lakh"].mean()
max_price = df["Price_Lakh"].max()
avg_area = df["Area_sqft"].mean()
house_count = len(df)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Average Price",
    f"₹{avg_price:.2f} Lakh"
)

col2.metric(
    "Maximum Price",
    f"₹{max_price:.2f} Lakh"
)

col3.metric(
    "Average Area",
    f"{avg_area:.0f} sqft"
)

col4.metric(
    "Properties",
    house_count
)

st.divider()


# ==========================================
# TABS
# ==========================================

tabs = st.tabs([
    "🏠 Price Prediction",
    "📊 Market Analysis",
    "🤖 Model Performance"
])

prediction_tab, analysis_tab, model_tab = tabs


# =========================================================
# PREDICTION TAB
# =========================================================

with prediction_tab:

    st.subheader("🏡 Property Information")

    col1, col2 = st.columns(2)


    # ==========================================
    # PROPERTY DETAILS
    # ==========================================

    with col1:

        st.markdown("#### 📐 Property Details")

        area = st.number_input(
            "Area (sqft)",
            min_value=100,
            max_value=10000,
            value=1200
        )

        bedrooms = st.number_input(
            "Bedrooms",
            min_value=1,
            max_value=10,
            value=3
        )

        bathrooms = st.number_input(
            "Bathrooms",
            min_value=1,
            max_value=10,
            value=2
        )

        age = st.number_input(
            "Property Age (years)",
            min_value=0,
            max_value=100,
            value=10
        )


    # ==========================================
    # LOCATION DETAILS
    # ==========================================

    with col2:

        st.markdown("#### 📍 Location Details")

        distance = st.number_input(
            "Distance to City (km)",
            min_value=0,
            max_value=100,
            value=10
        )

        location = st.selectbox(
            "Location",
            ["Noida", "Ghaziabad", "Delhi", "Gurugram"]
        )

        parking = st.selectbox(
            "Parking",
            ["Yes", "No"]
        )


    st.divider()


    # ==========================================
    # CREATE INPUT DATAFRAME
    # ==========================================

    input_data = pd.DataFrame({

        "Area_sqft": [area],

        "Bedrooms": [bedrooms],

        "Bathrooms": [bathrooms],

        "Age_Years": [age],

        "Distance_to_City_km": [distance],

        "Location": [location],

        "Parking": [parking]

    })


    # ==========================================
    # ENCODE CATEGORICAL FEATURES
    # ==========================================

    input_encoded = pd.get_dummies(
        input_data,
        columns=[
            "Location",
            "Parking"
        ],
        drop_first=False
    )


    # ==========================================
    # MATCH TRAINING FEATURES
    # ==========================================

    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # ==========================================
    # DISPLAY HOUSE SUMMARY
    # ==========================================

    st.subheader("🏡 House Summary")

    display_data = pd.DataFrame({

        "Feature": [
            "Area",
            "Bedrooms",
            "Bathrooms",
            "Property Age",
            "Distance to City",
            "Location",
            "Parking"
        ],

        "Value": [
            f"{area} sqft",
            bedrooms,
            bathrooms,
            f"{age} years",
            f"{distance} km",
            location,
            parking
        ]

    })


    st.dataframe(
        display_data,
        hide_index=True,
        width="stretch"
    )


    # ==========================================
    # PREDICT PRICE
    # ==========================================

    if st.button("🔮 Predict House Price"):

        prediction = model.predict(input_encoded)[0]

        st.divider()

        st.subheader("💰 Prediction Result")


        # ==========================================
        # PRICE COMPARISON
        # ==========================================

        if prediction > avg_price:

            price_status = "📈 Above average market price"

        elif prediction < avg_price:

            price_status = "📉 Below average market price"

        else:

            price_status = "➡️ Around the market average"


        result_col1, result_col2 = st.columns([2, 1])


        with result_col1:

            st.metric(
                "Estimated House Price",
                f"₹{prediction:.2f} Lakh",
                delta=f"₹{prediction - avg_price:.2f} Lakh vs average"
            )

            st.write(price_status)


        with result_col2:

            st.metric(
                "Market Average",
                f"₹{avg_price:.2f} Lakh"
            )


        st.divider()


        # ==========================================
        # PROPERTY SUMMARY
        # ==========================================

        st.markdown("### 🏡 Property Summary")

        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

        summary_col1.metric(
            "Area",
            f"{area} sqft"
        )

        summary_col2.metric(
            "Bedrooms",
            bedrooms
        )

        summary_col3.metric(
            "Bathrooms",
            bathrooms
        )

        summary_col4.metric(
            "Location",
            location
        )


# =========================================================
# MARKET ANALYSIS TAB
# =========================================================

with analysis_tab:

    st.subheader("📊 Market Analysis")

    st.write(
        "Explore house prices and their relationship "
        "with property features."
    )


    # ==========================================
    # LOCATION PRICE INSIGHT
    # ==========================================

    st.markdown("### 📍 Location Price Insight")

    selected_location = st.selectbox(
        "Select a location",
        sorted(df["Location"].unique())
    )

    location_data = df[
        df["Location"] == selected_location
    ]

    location_avg = location_data["Price_Lakh"].mean()

    difference = location_avg - avg_price

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Selected Location",
            selected_location
        )

    with col2:

        st.metric(
            "Average Price",
            f"₹{location_avg:.2f} Lakh"
        )

    with col3:

        st.metric(
            "vs Market Average",
            f"₹{difference:.2f} Lakh"
        )


    st.divider()


    # ==========================================
    # LOCATION PRICE RANGE
    # ==========================================

    st.markdown("### 🏘️ Location Price Range")

    min_price = location_data["Price_Lakh"].min()

    max_location_price = location_data["Price_Lakh"].max()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Lowest Price",
            f"₹{min_price:.2f} Lakh"
        )

    with col2:

        st.metric(
            "Highest Price",
            f"₹{max_location_price:.2f} Lakh"
        )


    st.divider()


    # ==========================================
    # CHART 1 & 2
    # ==========================================

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 💰 Price Distribution")

        st.image(
            "charts/price_distribution.png",
            width="stretch"
        )


    with col2:

        st.markdown("### 📐 Area vs Price")

        st.image(
            "charts/area_vs_price.png",
            width="stretch"
        )


    st.divider()


    # ==========================================
    # CHART 3 & 4
    # ==========================================

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 📍 Average Price by Location")

        st.image(
            "charts/price_by_location.png",
            width="stretch"
        )


    with col2:

        st.markdown("### 🛏️ Average Price by Bedrooms")

        st.image(
            "charts/price_by_bedrooms.png",
            width="stretch"
        )


# =========================================================
# MODEL PERFORMANCE TAB
# =========================================================

with model_tab:

    st.subheader("🤖 Model Performance")

    st.write(
        "Performance metrics and prediction accuracy "
        "of the Linear Regression model."
    )


    # ==========================================
    # LOAD MODEL METRICS
    # ==========================================

    with open(
        "model/evaluation_report.txt",
        "r"
    ) as file:

        report = file.read()


    metrics = {}

    for line in report.splitlines():

        if ":" in line:

            name, value = line.split(":", 1)

            try:

                metrics[name.strip()] = float(
                    value.strip()
                )

            except ValueError:

                pass


    mae_value = metrics.get("MAE", 0)

    mse_value = metrics.get("MSE", 0)

    rmse_value = metrics.get("RMSE", 0)

    r2_value = metrics.get("R2 Score", 0)


    # ==========================================
    # PERFORMANCE METRICS
    # ==========================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "MAE",
            f"{mae_value:.2f}"
        )

    with col2:

        st.metric(
            "MSE",
            f"{mse_value:.2f}"
        )

    with col3:

        st.metric(
            "RMSE",
            f"{rmse_value:.2f}"
        )

    with col4:

        st.metric(
            "R² Score",
            f"{r2_value:.3f}"
        )


    st.divider()


    # ==========================================
    # ACTUAL VS PREDICTED
    # ==========================================

    st.markdown(
        "### 🎯 Actual vs Predicted Prices"
    )

    st.image(
        "charts/actual_vs_predicted.png",
        width="stretch"
    )


    st.divider()


    # ==========================================
    # MODEL INFORMATION
    # ==========================================

    st.markdown("### 🧠 Model Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Algorithm",
            "Linear Regression"
        )

    with col2:

        st.metric(
            "Task",
            "Regression"
        )

    with col3:

        st.metric(
            "Target",
            "Price (Lakh)"
        )