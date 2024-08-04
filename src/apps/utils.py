import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

def load_data(uploaded_file):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        return df
    return None

# def preprocess_data(df, date_col, target_col, date_format, frequency):
#     df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors='coerce')
#     df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
#     df = df.dropna(subset=[date_col, target_col])
#     df = df[[date_col, target_col]].groupby(date_col).sum().reset_index()
#     df.set_index(date_col, inplace=True)
#     df = df.asfreq(freq=frequency)
#     df[target_col] = df[target_col].fillna(df[target_col].rolling(window=3, min_periods=1).mean())
#     return df

def preprocess_data(df, date_col, target_col, date_format, frequency):
    df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors='coerce', yearfirst=True)
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[date_col, target_col])
    df = df[[date_col, target_col]].groupby(date_col).sum().reset_index()
    df.set_index(date_col, inplace=True)
    df = df.resample(frequency).sum()
    df[target_col] = df[target_col].fillna(df[target_col].rolling(window=3, min_periods=1).mean())
    return df

def train_test_split(df, test_size):
    n = len(df)
    train_df = df.iloc[:int(n*(1-test_size))]
    test_df = df.iloc[int(n*(1-test_size)):]
    return train_df, test_df

def sarimax_forecast(train, test, order, seasonal_order):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    return forecast

def holt_winters_forecast(train, test, trend, seasonal, seasonal_periods):
    model = ExponentialSmoothing(train, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    return model_fit, forecast

def calculate_kpis(test, sarimax_forecast, hw_forecast):
    metrics = {
        "MAE": [mean_absolute_error(test, sarimax_forecast), mean_absolute_error(test, hw_forecast)],
        "MAPE": [mean_absolute_percentage_error(test, sarimax_forecast), mean_absolute_percentage_error(test, hw_forecast)],
        "R2": [r2_score(test, sarimax_forecast), r2_score(test, hw_forecast)]
    }
    return pd.DataFrame(metrics, index=["SARIMAX", "Holt-Winters"])




