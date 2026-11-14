
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
from datetime import date

# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="NightWatch AI",
    page_icon="🌙",
    layout="wide"
)

# -----------------------------------------
# LOAD PROJECT FILES
# -----------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("nightwatch_model.joblib")

@st.cache_data
def load_data():
    historical = pd.read_csv("historical_data.csv")
    anomalies = pd.read_csv("anomaly_data.csv")
    performance = pd.read_csv("model_performance.csv")
    importance = pd.read_csv("feature_importance.csv")
    cities = pd.read_csv("city_anomalies.csv", index_col=0)
    monthly = pd.read_csv("monthly_anomalies.csv", index_col=0)

    with open("metadata.json") as f:
        metadata = json.load(f)

    historical["date"] = pd.to_datetime(historical["date"])
    anomalies["date"] = pd.to_datetime(anomalies["date"])

    return historical, anomalies, performance, importance, cities, monthly, metadata

model = load_model()

historical, anomalies, performance, importance, cities, monthly, metadata = load_data()

# -----------------------------------------
# HEADER
# -----------------------------------------

st.title("🌙 NightWatch AI")
st.subheader("AI-powered nighttime temperature prediction and warm anomaly detection")

st.markdown(
    "Monitoring nighttime temperature patterns across five UAE cities "
    "using machine learning."
)

st.divider()

# -----------------------------------------
# MODEL PERFORMANCE
# -----------------------------------------

st.header("Model Performance")

xgb_row = performance[
    performance["Model"].str.contains("XGBoost", case=False)
]

if not xgb_row.empty:
    mae = xgb_row.iloc[0]["MAE (°C)"]
    rmse = xgb_row.iloc[0]["RMSE (°C)"]
    r2 = xgb_row.iloc[0]["R²"]

    col1, col2, col3 = st.columns(3)

    col1.metric("Mean Absolute Error", f"{mae:.3f} °C")
    col2.metric("Root Mean Squared Error", f"{rmse:.3f} °C")
    col3.metric("R² Score", f"{r2:.3f}")

st.caption("Performance metrics are based on the held-out evaluation dataset.")

st.divider()

# -----------------------------------------
# TEMPERATURE PREDICTION
# -----------------------------------------

st.header("Predict Nighttime Temperature")

st.write(
    "Enter daytime weather measurements and recent nighttime temperatures "
    "to estimate the upcoming night's average temperature."
)

cities_list = metadata["cities"]

city = st.selectbox("Select UAE city", cities_list)

city_history = historical[historical["city"] == city]

# Use city-specific historical medians as sensible starting values
defaults = city_history.median(numeric_only=True)

with st.form("prediction_form"):

    selected_date = st.date_input(
        "Prediction date",
        value=date.today()
    )

    col1, col2 = st.columns(2)

    with col1:
        daytime_avg_temp = st.number_input(
            "Daytime average temperature (°C)",
            value=float(defaults["daytime_avg_temp"])
        )

        daytime_max_temp = st.number_input(
            "Daytime maximum temperature (°C)",
            value=float(defaults["daytime_max_temp"])
        )

        daytime_avg_humidity = st.number_input(
            "Average humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(defaults["daytime_avg_humidity"])
        )

        daytime_avg_wind = st.number_input(
            "Average wind speed (km/h)",
            min_value=0.0,
            value=float(defaults["daytime_avg_wind"])
        )

    with col2:
        daytime_avg_cloud = st.number_input(
            "Average cloud cover (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(defaults["daytime_avg_cloud"])
        )

        previous_night_temp = st.number_input(
            "Previous night's average temperature (°C)",
            value=float(defaults["previous_night_temp"])
        )

        two_nights_ago_temp = st.number_input(
            "Temperature two nights ago (°C)",
            value=float(defaults["two_nights_ago_temp"])
        )

        previous_3_night_avg = st.number_input(
            "Previous 3-night average temperature (°C)",
            value=float(defaults["previous_3_night_avg"])
        )

    submitted = st.form_submit_button("Predict Nighttime Temperature")

if submitted:

    input_data = pd.DataFrame([{
        "city": city,
        "daytime_avg_temp": daytime_avg_temp,
        "daytime_max_temp": daytime_max_temp,
        "daytime_avg_humidity": daytime_avg_humidity,
        "daytime_avg_wind": daytime_avg_wind,
        "daytime_avg_cloud": daytime_avg_cloud,
        "month": selected_date.month,
        "day_of_year": selected_date.timetuple().tm_yday,
        "previous_night_temp": previous_night_temp,
        "two_nights_ago_temp": two_nights_ago_temp,
        "previous_3_night_avg": previous_3_night_avg
    }])

    # Preserve the exact training feature order
    input_data = input_data[metadata["input_features"]]

    prediction = float(model.predict(input_data)[0])

    st.success("Prediction generated successfully!")

    st.metric(
        "Predicted Nighttime Average Temperature",
        f"{prediction:.2f} °C"
    )

    st.info(
        "This is a model prediction, not a confirmed weather observation. "
        "A warm anomaly can only be confirmed after the actual nighttime "
        "temperature is observed and compared with the prediction."
    )

st.divider()

# -----------------------------------------
# HISTORICAL TEMPERATURE TRENDS
# -----------------------------------------

st.header("Historical Nighttime Temperature Trends")

selected_city = st.selectbox(
    "Choose a city for historical analysis",
    cities_list,
    key="historical_city"
)

city_history = historical[
    historical["city"] == selected_city
].sort_values("date")

fig = px.line(
    city_history,
    x="date",
    y="night_avg_temp",
    title=f"Historical Nighttime Temperature: {selected_city}",
    labels={
        "date": "Date",
        "night_avg_temp": "Average Night Temperature (°C)"
    }
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------
# WARM ANOMALY ANALYSIS
# -----------------------------------------

st.header("Warm Night Anomaly Detection")

st.write(
    "These results show unusually warm nights detected in the historical "
    "evaluation period."
)

city_results = cities.reset_index()
city_results = city_results.rename(
    columns={city_results.columns[0]: "City"}
)

fig = px.bar(
    city_results,
    x="City",
    y=["Global_Anomalies", "City_Adjusted_Anomalies"],
    barmode="group",
    title="Warm Anomalies by City",
    labels={
        "value": "Number of anomalies",
        "variable": "Detection method"
    }
)

st.plotly_chart(fig, use_container_width=True)

monthly_results = monthly.copy().reset_index()
monthly_results = monthly_results.rename(
    columns={monthly_results.columns[0]: "Month"}
)

fig = px.bar(
    monthly_results,
    x="Month",
    y="Warm_Anomalies",
    title="Warm Anomalies by Month",
    labels={
        "Warm_Anomalies": "Number of anomalies",
        "Month": "Month"
    }
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------
# FEATURE IMPORTANCE
# -----------------------------------------

st.header("What Influences the Predictions?")

feature_chart = importance.sort_values(
    "Importance", ascending=True
).tail(10)

fig = px.bar(
    feature_chart,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top 10 Model Features"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------
# HISTORICAL ANOMALY RECORDS
# -----------------------------------------

st.header("Detected Warm Night Records")

city_anomalies = anomalies[
    (anomalies["city"] == selected_city) &
    (anomalies["warm_anomaly"] == True)
].sort_values("prediction_error", ascending=False)

if not city_anomalies.empty:

    st.dataframe(
        city_anomalies[
            ["date", "night_avg_temp", "predicted_temp", "prediction_error"]
        ],
        use_container_width=True
    )

else:
    st.write("No global-threshold warm anomalies recorded for this city.")

# -----------------------------------------
# FOOTER
# -----------------------------------------

st.divider()

st.caption(
    "NightWatch AI | UAE nighttime temperature analysis | "
    "Machine-learning research prototype"
)
