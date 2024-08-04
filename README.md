# 📈 Time Series Forecasting App

## Description

This Time Series Forecasting App is built using Streamlit, providing users with an intuitive interface to perform time series forecasting using SARIMAX and Holt-Winters (Triple Exponential Smoothing) methods. Users can upload their time series data, configure model parameters, and visualize forecasts along with diagnostics.

## What is Time Series Forecasting?

Time series forecasting involves predicting future values based on previously observed values in a time-ordered sequence. This technique is essential in various fields such as finance, economics, weather forecasting, and supply chain management.

## Holt-Winters Method (Triple Exponential Smoothing)

The Holt-Winters method is a popular time series forecasting technique that applies exponential smoothing to capture trends and seasonality. It consists of three components:

1. **Level (L)**: The average value in the series.
2. **Trend (T)**: The increasing or decreasing value in the series.
3. **Seasonality (S)**: The repeating short-term cycle in the series.

Triple Exponential Smoothing is used to forecast data with both trend and seasonal patterns. It adjusts the level, trend, and seasonality components iteratively to generate accurate forecasts.

## SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors)

SARIMAX is an extension of the ARIMA model that supports univariate time series data with a trend and seasonal components. The model is defined by parameters (p, d, q) for the ARIMA part and (P, D, Q, s) for the seasonal part:

- **p**: Number of lag observations included in the model (AR part).
- **d**: Number of times that the raw observations are differenced (I part).
- **q**: Size of the moving average window (MA part).
- **P, D, Q**: The seasonal counterparts of the ARIMA parameters.
- **s**: The periodicity (number of time steps for a single seasonal cycle).

SARIMAX is used for forecasting time series data that exhibit both non-seasonal and seasonal patterns.

## Features

- **📤 Upload and Configure Data**: Users can upload their CSV files, select date and target columns, specify date format and frequency.
- **⚙️ Model Configuration**: Allows users to configure SARIMAX and Holt-Winters model parameters.
- **📊 Visualizations**: Provides interactive plots for the entire dataset, autocorrelation, forecasts, and model diagnostics.
- **📈 KPIs**: Displays key performance indicators for model evaluation.

## How to Use

1. **Upload Data**: Upload your CSV file containing the time series data.
2. **Configure Columns**: Select the date and target columns, specify the date format and frequency.
3. **Submit**: Click the "Submit" button to process the data.
4. **Configure Models**: Adjust the parameters for SARIMAX and Holt-Winters models using the sidebar controls.
5. **View Forecasts and Diagnostics**: Explore the forecasts, decomposition, and diagnostics plots.
6. **Evaluate Performance**: Review the KPIs to evaluate model performance.
