import streamlit as st
from apps.utils import load_data, preprocess_data, train_test_split, sarimax_forecast, holt_winters_forecast, calculate_kpis
from apps.visualizations import plot_forecasts, plot_decomposition, plot_mape,plot_acf_pacf,plot_entire_data

def app():
    st.title("Time Series Forecasting")

    st.sidebar.header("Upload and Configure Data")
    st.sidebar.markdown("[Download Sample Datasets](https://github.com/dibyendutapadar/time-series-forecasting/tree/cd2f848fce49ece37a0dc3235449532075d9c9bf/sample%20data%20sets)")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)

        if df is not None:
            st.subheader("Data Preview")
            st.write(df.head(2))

            date_col = st.sidebar.selectbox("Select Date Column", df.columns)
            target_col = st.sidebar.selectbox("Select Target Column", df.columns)
            date_format = st.sidebar.text_input("Date Format (e.g., %Y-%m-%d)", value="%Y-%m-%d")
            frequency = st.sidebar.selectbox("Select Frequency", ["D", "W", "M", "Q", "Y"])

            if st.sidebar.button("Submit"):
                df = preprocess_data(df, date_col, target_col, date_format, frequency)
                st.session_state['df'] = df
                st.session_state['date_col'] = date_col
                st.session_state['target_col'] = target_col

    if 'df' in st.session_state:
        df = st.session_state['df']
        date_col = st.session_state['date_col']
        target_col = st.session_state['target_col']




        st.subheader("Entire Data Plot")
        fig_entire_data = plot_entire_data(df, date_col, target_col)
        st.plotly_chart(fig_entire_data)

        st.subheader("ACF and PACF Plots")
        st.plotly_chart(plot_acf_pacf(df[target_col]))



        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

        st.header("SARIMAX Model")
        with st.expander("SARIMAX Models Explanation"):
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


        st.subheader("Configure")

        # Create the first row with 3 columns
        col1, col2, col3 = st.columns(3)

        with col1:
            p = st.number_input("Order p", min_value=0, max_value=10, value=1)

        with col2:
            d = st.number_input("Order d", min_value=0, max_value=10, value=1)

        with col3:
            q = st.number_input("Order q", min_value=0, max_value=10, value=1)
        # Create the second row with 5 columns
        col4, col5, col6, col7 = st.columns(4)

        with col4:
            sp = st.number_input("Seasonal Order p", min_value=0, max_value=10, value=1)

        with col5:
            sd = st.number_input("Seasonal Order d", min_value=0, max_value=10, value=1)

        with col6:
            sq = st.number_input("Seasonal Order q", min_value=0, max_value=10, value=1)

        with col7:
            s = st.number_input("Seasonal Period s", min_value=1, max_value=365, value=12)



        train, test = train_test_split(df, test_size)
        sarimax_pred = sarimax_forecast(train[target_col], test[target_col], (p, d, q), (sp, sd, sq, s))


        st.subheader("SARIMAX Forecast")
        st.plotly_chart(plot_forecasts(train[target_col], test[target_col], sarimax_pred, method="SARIMAX"))
       
        st.markdown("---")


        st.header("Holt-Winters Model")



        with st.expander("Holt-Winters Triple Exponential Smoothing"):
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


        st.subheader("Configure")

        col8, col9, col10 = st.columns(3)

        with col8:
            trend = st.selectbox("Select Trend", [None, 'add', 'mul'])
        
        with col9:
            seasonal = st.selectbox("Select Seasonal", [None, 'add', 'mul'])

        with col10:    
            seasonal_periods = st.number_input("Seasonal Periods", min_value=1, max_value=365, value=12)
        
        hw_model, hw_pred = holt_winters_forecast(train[target_col], test[target_col], trend, seasonal, seasonal_periods)

        st.subheader("Holt-Winters Decomposition")

        st.pyplot(plot_decomposition(train[target_col], seasonal_periods))

        st.subheader("Holt-Winters Forecast")
        st.plotly_chart(plot_forecasts(train[target_col], test[target_col], hw_pred, method="Holt-Winters"))

        st.subheader("KPIs")
        kpis = calculate_kpis(test[target_col], sarimax_pred, hw_pred)
        st.write(kpis)

        st.subheader("MAPE Comparison")
        st.plotly_chart(plot_mape(kpis))
