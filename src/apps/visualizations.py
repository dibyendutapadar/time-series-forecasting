import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.gofplots import qqplot
import plotly.express as px



def plot_entire_data(df, date_col, target_col):
    df = df.reset_index() 
    fig = px.line(df, x=date_col, y=target_col, title='Entire Data Plot')
    fig.update_xaxes(rangeslider_visible=True)
    
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=2, label="2y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    return fig

def plot_forecasts(train, test, forecast, method):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=train.index, y=train, mode='lines', name='Train'))
    fig.add_trace(go.Scatter(x=test.index, y=test, mode='lines', name='Test'))
    fig.add_trace(go.Scatter(x=test.index, y=forecast, mode='lines', name=f'{method} Forecast'))
    
    fig.update_layout(title=f"{method} Forecast", xaxis_title='Date', yaxis_title='Value')
    return fig

def plot_decomposition(train, seasonal_periods):
    decomposition = seasonal_decompose(train, model='additive', period=seasonal_periods)
    fig, axes = plt.subplots(4, 1, figsize=(15, 8), sharex=True)
    
    decomposition.observed.plot(ax=axes[0], legend=False)
    axes[0].set_ylabel('Observed')
    decomposition.trend.plot(ax=axes[1], legend=False)
    axes[1].set_ylabel('Trend')
    decomposition.seasonal.plot(ax=axes[2], legend=False)
    axes[2].set_ylabel('Seasonal')
    decomposition.resid.plot(ax=axes[3], legend=False)
    axes[3].set_ylabel('Residual')
    plt.tight_layout()
    return fig

def plot_mape(kpis):
    fig = go.Figure(data=[
        go.Bar(name='SARIMAX', x=['MAPE'], y=[kpis.loc['SARIMAX', 'MAPE']]),
        go.Bar(name='Holt-Winters', x=['MAPE'], y=[kpis.loc['Holt-Winters', 'MAPE']])
    ])
    
    fig.update_layout(barmode='group', title='MAPE Comparison', xaxis_title='KPI', yaxis_title='Value')
    return fig

def plot_acf_pacf(series):
    # Calculate ACF and PACF
    acf_vals = acf(series, nlags=50)
    pacf_vals = pacf(series, nlags=50)
    
    # Create subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=('ACF', 'PACF'))
    
    # ACF Plot
    fig.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name='ACF'), row=1, col=1)
    
    # PACF Plot
    fig.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, name='PACF'), row=1, col=2)
    
    fig.update_layout(title_text='ACF and PACF Plots', height=400)
    
    return fig

def plot_sarimax_diagnostics(model_fit):
    residuals = model_fit.resid
    
    # Residuals plot
    fig_residuals = go.Figure()
    fig_residuals.add_trace(go.Scatter(y=residuals, mode='lines', name='Residuals'))
    fig_residuals.update_layout(title='Residuals', xaxis_title='Time', yaxis_title='Residuals')
    
    # Q-Q plot
    qq = qqplot(residuals, line='s')
    qq_line = qq.gca().lines[1]
    qq_fig = go.Figure()
    qq_fig.add_trace(go.Scatter(x=qq_line.get_xdata(), y=qq_line.get_ydata(), mode='lines', name='Q-Q Line'))
    qq_fig.add_trace(go.Scatter(x=qq_line.get_xdata(), y=qq_line.get_ydata(), mode='markers', name='Sample Data'))
    qq_fig.update_layout(title='Q-Q Plot', xaxis_title='Theoretical Quantiles', yaxis_title='Sample Quantiles')
    
    # ACF plot
    acf_vals = acf(residuals, nlags=50)
    fig_acf = go.Figure()
    fig_acf.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name='ACF'))
    fig_acf.update_layout(title='ACF', xaxis_title='Lags', yaxis_title='ACF')
    
    # PACF plot
    pacf_vals = pacf(residuals, nlags=50)
    fig_pacf = go.Figure()
    fig_pacf.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, name='PACF'))
    fig_pacf.update_layout(title='PACF', xaxis_title='Lags', yaxis_title='PACF')
    
    return fig_residuals, qq_fig, fig_acf, fig_pacf
