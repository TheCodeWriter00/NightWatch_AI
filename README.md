# NightWatch AI

Track: Climate
Team: NightWatch AI UAE

## The problem

High nighttime temperatures are an overlooked part of extreme heat, particularly in hot regions such as the UAE. Unusually warm nights can prevent the body from recovering from daytime heat exposure. NightWatch AI addresses this by predicting nighttime temperatures and analysing historical warm-temperature anomalies across five UAE cities.

## What we built

NightWatch AI is a Streamlit-based climate prediction dashboard that uses machine learning to forecast average nighttime temperatures for Abu Dhabi, Al Ain, Dubai, Sharjah, and Ras Al Khaimah.

It retrieves weather data automatically through the Open-Meteo API and generates three-night predictions using an XGBoost model. The dashboard also includes historical temperature trends, warm-anomaly detection, monthly anomaly analysis, and feature importance visualisations.

## What we added

Our own idea.

We developed our own approach by combining nighttime temperature prediction with historical anomaly analysis. We incorporated city-specific features, previous nighttime temperatures, automated weather retrieval, and extended forecasting from one night to three consecutive nights.

## How it works

We collected approximately 350,640 hourly weather records and processed them into 14,590 daily records. Features include daytime temperature, humidity, wind speed, cloud cover, previous nighttime temperatures, and calendar information.

We experimented with Persistence Baseline, Random Forest, XGBoost, and LSTM models. XGBoost was selected for the final system, achieving an MAE of 0.846°C, RMSE of 1.092°C, and R² of 0.965 on held-out test data.

The model uses historical patterns and automatically retrieved weather features to predict nighttime temperatures. For the second and third nights, previous model predictions are used as lag features. Historical anomalies are identified using city-specific thresholds.

## What we’d do next

With more development time, we would expand NightWatch AI beyond the UAE to cover more cities and regions. We would also extend forecasting from three nights to five or seven nights, while improving prediction reliability over longer periods.

## Project Links

* GitHub repository: https://github.com/TheCodeWriter00/NightWatch_AI/tree/main
* Live demo: https://night-watch-ai-uae.streamlit.app

## Sources

* Weather data: Open-Meteo Historical Weather API and Forecast API.
* Programming language: Python.
* Machine learning: XGBoost, Scikit-learn, PyTorch (LSTM), Random Forest.
* Data processing: Pandas, NumPy.
* Visualisation: Matplotlib, Plotly.
* Dashboard: Streamlit.
* Model storage: Joblib.
* API integration: Requests.
* Development and deployment: Google Colab, GitHub, Streamlit Community Cloud.
* Development assistance: ChatGPT, used for occasional programming explanations.
* Model evaluation: Persistence Baseline, Random Forest, XGBoost, and PyTorch LSTM.
