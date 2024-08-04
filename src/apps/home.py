import streamlit as st

def app():
    st.title("Welcome to Time Series Forecasting App")
    st.write("""
    This app allows you to upload time series data and forecast using ARIMA and Holt-Winters (Triple Exponential Smoothing) methods.
    Navigate to the Forecast page to get started.
    """)
    st.header("Holt-Winters Triple Exponential Smoothing")
    st.markdown("""
    Holt-Winters Triple Exponential Smoothing, also known as Holt-Winters Method, is used for forecasting time series data that exhibits both trend and seasonality. The model requires you to set values for trend, seasonal, and seasonal_periods. Here’s a guide on how to determine these values:

    1. **Understanding the Parameters**:
    - **Trend (`trend`)**: This parameter indicates whether a trend component is included in the model. It can be:
        - `'add'` (additive): When the trend is expected to increase or decrease by a constant amount.
        - `'mul'` (multiplicative): When the trend increases or decreases by a percentage or ratio.
        - `None`: When there is no trend.
    - **Seasonal (`seasonal`)**: This parameter indicates whether a seasonal component is included in the model. It can be:
        - `'add'` (additive): When seasonal variations are roughly constant over time.
        - `'mul'` (multiplicative): When seasonal variations change proportionally to the level of the time series.
        - `None`: When there is no seasonality.
    - **Seasonal Periods (`seasonal_periods`)**: This parameter indicates the length of the seasonality cycle (e.g., 12 for monthly data with annual seasonality).

    2. **Selecting the Parameters**:
    - **a. Identify Trend and Seasonality**:
        - **Visual Inspection**: Plot your time series data. Look for patterns in the data:
            - A consistent upward or downward slope suggests a trend.
            - Repeating patterns at regular intervals suggest seasonality.
        - **Statistical Tests**: Use autocorrelation plots (ACF) and partial autocorrelation plots (PACF) to identify seasonality. Significant spikes at lags corresponding to the seasonal period can indicate seasonality.
    - **b. Determine Seasonal Periods**:
        - **Domain Knowledge**: Use domain knowledge to identify the seasonal cycle. For instance, monthly data often has an annual seasonality period of 12 months.
        - **Autocorrelation**: Significant autocorrelations at specific lags can help identify the seasonal period.
    - **c. Choosing Trend and Seasonal Types**:
        - **Additive vs. Multiplicative**:
            - Use additive (`'add'`) if the seasonal variations are roughly constant over time (the difference between high and low seasonality periods is constant).
            - Use multiplicative (`'mul'`) if the seasonal variations change proportionally with the level of the series (the percentage change is constant).
    """)


    st.header("SARIMAX Model")
    st.markdown("""
    SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors) is an extension of the ARIMA model that supports both seasonality and exogenous variables. It is used for forecasting time series data. The model parameters need to be carefully chosen to fit the data well. Here’s a guide on understanding and selecting these parameters:

    1. **Understanding the Parameters**:
    - **Non-Seasonal Components**:
        - **p**: The number of lag observations included in the model (autoregressive part).
        - **d**: The number of times that the raw observations are differenced (integrated part).
        - **q**: The size of the moving average window.
    - **Seasonal Components**:
        - **P**: The number of lag observations in the seasonal model (seasonal autoregressive part).
        - **D**: The number of times that the seasonal differences are applied (seasonal integrated part).
        - **Q**: The size of the moving average window in the seasonal model.
        - **m**: The number of time steps for a single seasonal period.
    - **Exogenous Variables (X)**: These are additional variables that can be included in the model to provide more information that might affect the time series.

    2. **Selecting the Parameters**:
    - **a. Identify Non-Seasonal Parameters (p, d, q)**:
        - **Visual Inspection**: Plot your time series data to look for trends and seasonality.
        - **ACF and PACF Plots**:
            - **ACF (Autocorrelation Function)**: Helps in identifying the number of MA (q) terms. Significant spikes outside the confidence interval suggest the lag values.
            - **PACF (Partial Autocorrelation Function)**: Helps in identifying the number of AR (p) terms. Significant spikes outside the confidence interval suggest the lag values.
        - **Differencing**:
            - If the time series is non-stationary, apply differencing and observe the ACF and PACF plots again. Differencing helps to stabilize the mean of the time series by removing changes in the level of a time series, thereby eliminating (or reducing) trend and seasonality.

    - **b. Identify Seasonal Parameters (P, D, Q, m)**:
        - **Seasonal Differencing**: If seasonality is present, seasonal differencing may be necessary.
        - **Seasonal ACF and PACF**:
            - Look at the ACF and PACF plots of the seasonally differenced series to identify seasonal AR (P) and MA (Q) terms.
            - The seasonal period (m) should be known from the data (e.g., m = 12 for monthly data with yearly seasonality).

    - **c. Combining the Parameters**:
        - Use the identified non-seasonal and seasonal parameters to build the SARIMAX model.
        - **Exogenous Variables**: Include any relevant external variables (X) if they are believed to influence the time series.
                        """)
