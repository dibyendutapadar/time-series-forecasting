import streamlit as st
from multiapp import MultiApp
from apps import home, forecast



st.set_page_config(
    page_title="Time Series Forecasting App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

app = MultiApp()

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Forecast", forecast.app)

# The main app
app.run()
