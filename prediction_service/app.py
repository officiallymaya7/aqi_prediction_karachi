# prediction_service/app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import hopsworks
import plotly.express as px
from dotenv import load_dotenv

# --- Configuration and Loading ---
load_dotenv() # Load secrets from .env file for local running

# Get the path to the model file
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_aqi_predictor.pkl')

@st.cache_resource
def load_model_and_features():
    """Loads the trained model and its feature names."""
    try:
        model = joblib.load(model_path)
        feature_names = model.feature_names_in_
        return model, feature_names
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please run the training script first.")
        return None, None

@st.cache_data(ttl=3600) # Cache data for 1 hour
def fetch_historical_data_for_plotting():
    """Fetches the last 24 hours of data from Hopsworks."""
    try:
        project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
        fs = project.get_feature_store()
        feature_group = fs.get_feature_group(name=os.getenv("FEATURE_GROUP_NAME"), version=1)
        
        # Read the feature group data
        df = feature_group.read()
        
        # Convert datetime column and sort
        df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
        df = df.sort_values('datetime')
        
        # Get the last 24 hours of data
        last_24h_data = df.tail(24)
        return last_24h_data
    except Exception as e:
        st.error(f"Could not fetch historical data from Hopsworks: {e}")
        return pd.DataFrame()

# --- Prediction Function ---
def predict_aqi(model, input_data):
    """Makes a prediction on the input data."""
    prediction = model.predict(input_data)
    return prediction[0]

# --- Main App UI ---
def main():
    st.set_page_config(page_title="Karachi AQI Predictor", page_icon="🌫️", layout="wide")
    
    st.title("🏢 Karachi Air Quality Index (AQI) Predictor")
    st.markdown("An interactive dashboard to predict and analyze air quality in Karachi.")

    # Load the model and feature names
    model, feature_names = load_model_and_features()
    if model is None:
        st.stop()

    # --- Sidebar for User Input ---
    st.sidebar.header("Input Data for Prediction")
    
    hour = st.sidebar.slider("Hour of the day (0-23)", 0, 23, 12)
    dayofweek = st.sidebar.slider("Day of the week (0=Mon, 6=Sun)", 0, 6, 3)
    pm2_5 = st.sidebar.number_input("PM2.5 Level", min_value=0.0, value=50.0, step=1.0)
    pm10 = st.sidebar.number_input("PM10 Level", min_value=0.0, value=100.0, step=1.0)
    co = st.sidebar.number_input("CO Level", min_value=0.0, value=200.0, step=1.0)
    no = st.sidebar.number_input("NO Level", min_value=0.0, value=0.1, step=0.01)
    no2 = st.sidebar.number_input("NO2 Level", min_value=0.0, value=0.5, step=0.01)
    o3 = st.sidebar.number_input("O3 Level", min_value=0.0, value=120.0, step=1.0)

    # Prepare data for prediction
    input_dict = {
        'aqi_change_rate': 0.0, 'co': co, 'no': no, 'no2': no2, 'o3': o3, 'so2': 1.0,
        'pm2_5': pm2_5, 'pm10': pm10, 'nh3': 0.1, 'hour': hour, 'dayofweek': dayofweek, 'month': 11
    }
    input_df = pd.DataFrame([input_dict], columns=feature_names)

    # --- Prediction and Output ---
    if st.sidebar.button("Predict AQI"):
        with st.spinner("Predicting..."):
            prediction = predict_aqi(model, input_df)
        
        st.header("🔮 Prediction Result")
        st.metric(label="Predicted AQI for Next Hour", value=f"{prediction:.2f}")

        # Provide a textual interpretation
        if prediction <= 1:
            st.success("The air quality is **Good**. Enjoy your outdoor activities!")
        elif prediction <= 2:
            st.info("The air quality is **Fair**. It's acceptable for most people.")
        elif prediction <= 3:
            st.warning("The air quality is **Moderate**. Sensitive individuals may experience minor issues.")
        elif prediction <= 4:
            st.error("The air quality is **Poor**. Everyone may begin to experience health effects.")
        else:
            st.error("The air quality is **Very Poor**. Health warnings of emergency conditions.")

    # --- Dashboard with Charts ---
    st.header("📊 Air Quality Dashboard")
    
    # Fetch historical data
    historical_df = fetch_historical_data_for_plotting()

    if not historical_df.empty:
        # Chart 1: Historical AQI Trend
        st.subheader("Last 24 Hours AQI Trend")
        fig_trend = px.line(
            historical_df, 
            x='datetime', 
            y='aqi', 
            title='AQI Over Time',
            labels={'datetime': 'Time', 'aqi': 'AQI Value'},
            markers=True
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # Chart 2: Current Pollutant Levels
        st.subheader("Current Pollutant Levels")
        pollutants_df = pd.DataFrame({
            'Pollutant': ['PM2.5', 'PM10', 'CO', 'NO', 'NO2', 'O3'],
            'Level': [pm2_5, pm10, co, no, no2, o3]
        })
        fig_pollutants = px.bar(
            pollutants_df, 
            x='Pollutant', 
            y='Level', 
            title='Input Pollutant Concentrations',
            color='Pollutant'
        )
        st.plotly_chart(fig_pollutants, use_container_width=True)
    else:
        st.warning("Could not display charts due to missing historical data.")
        
     # --- Styled Footer ---
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: grey;">'
        'Made with ❤️ by <b>Maya Khurshid Anwar</b>, 10Pearls Intern'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()