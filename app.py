

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import requests

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="NightWatch AI",
    page_icon="🌙",
    layout="wide"
)

UAE_TIMEZONE = ZoneInfo("Asia/Dubai")

# -----------------------------------------
# UAE CITY COORDINATES
# -----------------------------------------

CITY_COORDINATES = {
    "Abu Dhabi": (24.4539, 54.3773),
    "Al Ain": (24.1917, 55.7606),
    "Dubai": (25.2048, 55.2708),
    "Ras Al Khaimah": (25.7895, 55.9432),
    "Sharjah": (25.3463, 55.4209)
}

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

    return (
        historical,
        anomalies,
        performance,
        importance,
        cities,
        monthly,
        metadata
    )


model = load_model()

(
    historical,
    anomalies,
    performance,
    importance,
    cities,
    monthly,
    metadata
) = load_data()

# -----------------------------------------
# AUTOMATIC WEATHER RETRIEVAL
# -----------------------------------------

@st.cache_data(ttl=1800)
def fetch_weather(city):

    latitude, longitude = CITY_COORDINATES[city]

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "wind_speed_10m,"
            "cloud_cover"
        ),
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "timezone": "Asia/Dubai",
        "past_days": 4,
        "forecast_days": 3
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:
        raise ValueError("Weather data was not returned by the API.")

    weather = pd.DataFrame(data["hourly"])

    weather["time"] = pd.to_datetime(weather["time"])

    return weather


# -----------------------------------------
# CALCULATE DAYTIME WEATHER FEATURES
# -----------------------------------------

def calculate_daytime_features(weather, target_date):

    start = pd.Timestamp(
        datetime.combine(target_date, datetime.min.time())
    ) + pd.Timedelta(hours=6)

    end = pd.Timestamp(
        datetime.combine(target_date, datetime.min.time())
    ) + pd.Timedelta(hours=18)

    daytime = weather[
        (weather["time"] >= start) &
        (weather["time"] <= end)
    ].copy()

    required_columns = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "cloud_cover"
    ]

    if len(daytime) != 13:
        raise ValueError(
            "The API did not return all 13 daytime observations. "
            "Please try again later."
        )

    if daytime[required_columns].isna().any().any():
        raise ValueError(
            "Some daytime weather measurements are missing."
        )

    return {
        "daytime_avg_temp": daytime["temperature_2m"].mean(),
        "daytime_max_temp": daytime["temperature_2m"].max(),
        "daytime_avg_humidity": daytime["relative_humidity_2m"].mean(),
        "daytime_avg_wind": daytime["wind_speed_10m"].mean(),
        "daytime_avg_cloud": daytime["cloud_cover"].mean()
    }


# -----------------------------------------
# CALCULATE NIGHTTIME HISTORY
# -----------------------------------------

def calculate_night_average(weather, night_date):
    night_date = pd.to_datetime(night_date).date()

    times = pd.to_datetime(weather["time"])
    hours = times.dt.hour

    start = pd.Timestamp(night_date) + pd.Timedelta(hours=21)
    end = start + pd.Timedelta(hours=8)

    mask = (times >= start) & (times <= end)
    night_temps = weather.loc[mask, "temperature_2m"].dropna()

    if len(night_temps) != 9:
        raise ValueError(
            f"Expected 9 nighttime observations for {night_date}, "
            f"but received {len(night_temps)}"
        )

    return float(night_temps.mean())


# -----------------------------------------
# BUILD ALL MODEL INPUT FEATURES
# -----------------------------------------


def prepare_prediction_features(
    weather, city, target_date, previous_night_values=None
):

    daytime = calculate_daytime_features(
        weather,
        target_date
    )

    if previous_night_values is None:
        previous_night_values = [
            calculate_night_average(
                weather,
                target_date - timedelta(days=1)
            ),
            calculate_night_average(
                weather,
                target_date - timedelta(days=2)
            ),
            calculate_night_average(
                weather,
                target_date - timedelta(days=3)
            )
        ]

    previous_night, two_nights_ago, three_nights_ago = (
        previous_night_values
    )

    previous_three_night_avg = np.mean([
        previous_night,
        two_nights_ago,
        three_nights_ago
    ])

    features = {
        "city": city,
        "daytime_avg_temp": daytime["daytime_avg_temp"],
        "daytime_max_temp": daytime["daytime_max_temp"],
        "daytime_avg_humidity": daytime["daytime_avg_humidity"],
        "daytime_avg_wind": daytime["daytime_avg_wind"],
        "daytime_avg_cloud": daytime["daytime_avg_cloud"],
        "month": target_date.month,
        "day_of_year": target_date.timetuple().tm_yday,
        "previous_night_temp": previous_night,
        "two_nights_ago_temp": two_nights_ago,
        "previous_3_night_avg": previous_three_night_avg
    }

    return features



# -----------------------------------------
# HEADER
# -----------------------------------------

st.title("🌙 NightWatch AI")

st.subheader(
    "AI-powered nighttime temperature prediction "
    "and warm anomaly detection"
)

st.markdown(
    "Monitoring nighttime temperature patterns across "
    "five UAE cities using machine learning."
)

st.divider()

# -----------------------------------------
# MODEL PERFORMANCE
# -----------------------------------------

st.header("Model Performance")

xgb_row = performance[
    performance["Model"].str.contains(
        "XGBoost",
        case=False
    )
]

if not xgb_row.empty:

    mae = xgb_row.iloc[0]["MAE (°C)"]
    rmse = xgb_row.iloc[0]["RMSE (°C)"]
    r2 = xgb_row.iloc[0]["R²"]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Mean Absolute Error",
        f"{mae:.3f} °C"
    )

    col2.metric(
        "Root Mean Squared Error",
        f"{rmse:.3f} °C"
    )

    col3.metric(
        "R² Score",
        f"{r2:.3f}"
    )

st.caption(
    "Performance metrics are based on the held-out evaluation dataset."
)

st.divider()

# -----------------------------------------
# AUTOMATIC NIGHTTIME PREDICTION
# -----------------------------------------

st.header("Automatic Nighttime Temperature Prediction")

st.write(
    "Select a UAE city. NightWatch AI will automatically retrieve "
    "hourly weather data, calculate the required model features "
    "and predict the nighttime average temperature."
)

cities_list = metadata["cities"]

available_cities = [
    city for city in cities_list
    if city in CITY_COORDINATES
]

selected_prediction_city = st.selectbox(
    "Select UAE city",
    available_cities
)

today = datetime.now(UAE_TIMEZONE).date()

st.info(
    f"Prediction night: {today.strftime('%d %B %Y')} "
    "| UAE local time"
)

if st.button(
    "🌙 Fetch Weather & Predict",
    use_container_width=True
):

    try:

        with st.spinner(
            "Fetching weather data and preparing model inputs..."
        ):

            weather = fetch_weather(
                selected_prediction_city
            )

            features = prepare_prediction_features(
                weather,
                selected_prediction_city,
                today
            )

            input_data = pd.DataFrame([features])

            # Preserve exact training feature order
            input_data = input_data[
                metadata["input_features"]
            ]

            prediction = float(
                model.predict(input_data)[0]
            )

        st.success(
            "Weather data retrieved and prediction generated!"
        )

        st.metric(
            "Predicted Nighttime Average Temperature",
            f"{prediction:.2f} °C"
        )

        # ---------------------------------
        # DISPLAY AUTOMATICALLY FETCHED INPUTS
        # ---------------------------------

        with st.expander(
            "View automatically retrieved weather inputs"
        ):

            st.write(
                "Daytime weather (06:00–18:00)"
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Average temperature",
                f"{features['daytime_avg_temp']:.2f} °C"
            )

            col2.metric(
                "Maximum temperature",
                f"{features['daytime_max_temp']:.2f} °C"
            )

            col1.metric(
                "Average humidity",
                f"{features['daytime_avg_humidity']:.2f}%"
            )

            col2.metric(
                "Average wind speed",
                f"{features['daytime_avg_wind']:.2f} km/h"
            )

            col1.metric(
                "Average cloud cover",
                f"{features['daytime_avg_cloud']:.2f}%"
            )

            st.write(
                "Previous nighttime temperatures (21:00–05:00)"
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Previous night",
                f"{features['previous_night_temp']:.2f} °C"
            )

            col2.metric(
                "Two nights ago",
                f"{features['two_nights_ago_temp']:.2f} °C"
            )

            st.metric(
                "Previous 3-night average",
                f"{features['previous_3_night_avg']:.2f} °C"
            )

        st.caption(
            "Weather inputs are retrieved from Open-Meteo. "
            "The provider's weather-model data is not guaranteed "
            "to represent direct weather-station observations."
        )

        st.info(
            "This is a machine-learning prediction, not a confirmed "
            "weather observation. A warm anomaly can only be confirmed "
            "after the actual nighttime temperature is observed and "
            "compared with the prediction."
        )

    except requests.RequestException as error:

        st.error(
            "Unable to retrieve weather data. "
            "Please check your connection and try again."
        )

        st.caption(str(error))

    except (ValueError, KeyError) as error:

        st.error(
            "The prediction could not be generated because "
            "some required weather data is unavailable."
        )

        st.caption(str(error))

    except Exception as error:

        st.error(
            "An unexpected error occurred while generating "
            "the prediction."
        )

        st.caption(str(error))


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

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------------
# WARM ANOMALY ANALYSIS
# -----------------------------------------

st.header("Warm Night Anomaly Detection")

st.write(
    "These results show unusually warm nights detected "
    "in the historical evaluation period."
)

city_results = cities.reset_index()

city_results = city_results.rename(
    columns={city_results.columns[0]: "City"}
)

fig = px.bar(
    city_results,
    x="City",
    y=[
        "Global_Anomalies",
        "City_Adjusted_Anomalies"
    ],
    barmode="group",
    title="Warm Anomalies by City",
    labels={
        "value": "Number of anomalies",
        "variable": "Detection method"
    }
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------------
# MONTHLY ANOMALIES
# -----------------------------------------

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

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------------
# FEATURE IMPORTANCE
# -----------------------------------------

st.header("What Influences the Predictions?")

feature_chart = importance.sort_values(
    "Importance",
    ascending=True
).tail(10)

fig = px.bar(
    feature_chart,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top 10 Model Features"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------------
# HISTORICAL ANOMALY RECORDS
# -----------------------------------------

st.header("Detected Warm Night Records")

city_anomalies = anomalies[
    (anomalies["city"] == selected_city) &
    (anomalies["warm_anomaly"] == True)
].sort_values(
    "prediction_error",
    ascending=False
)

if not city_anomalies.empty:

    st.dataframe(
        city_anomalies[
            [
                "date",
                "night_avg_temp",
                "predicted_temp",
                "prediction_error"
            ]
        ],
        use_container_width=True
    )

else:

    st.write(
        "No global-threshold warm anomalies recorded for this city."
    )

# -----------------------------------------
# FOOTER
# -----------------------------------------

st.divider()

st.caption(
    "NightWatch AI | UAE nighttime temperature analysis | "
    "Machine-learning research prototype"
)
