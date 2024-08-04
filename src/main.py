import streamlit as st
from multiapp import MultiApp
from apps import home, forecast

app = MultiApp()

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Forecast", forecast.app)

# The main app
app.run()
