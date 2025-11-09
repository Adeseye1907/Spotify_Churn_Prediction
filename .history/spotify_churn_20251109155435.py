import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings('ignore')
import plotly.express as px

ds = pd.read_csv('spotify_churn_dataset.csv')
st.markdown("<h1 style = 'color: #DD5746; text-align: center; font-size:60px; font-family: Monospace'>SPOTIFY CHURN PREDICTION </h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin:30px; color:#FFC470; text-align:center; font-family:Serif'>Build by Adeseye </h4>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html = True)

st.image('spotify.png')
st.divider()

st.markdown("<h2 style = 'color: #F7C566; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)
st.markdown('Customer retention is a critical metric for digital streaming platforms such as Spotify, where user engagement and subscription continuity directly determine profitability. Despite offering a vast music library and personalized recommendations, Spotify continues to face challenges in minimizing user churn—the rate at which subscribers discontinue their service. Several behavioral and service-related factors influence churn tendencies. User engagement indicators such as listening time, skip rate, and offline listening frequency provide insight into satisfaction levels and platform loyalty. Likewise, exposure to advertisements and subscription type (free vs. premium) significantly shape user experience and long-term retention. This study aims to examine the determinants of churn among Spotify users, with a focus on identifying patterns that predict disengagement. By analyzing listening behavior, ad interactions, and subscription preferences, the analysis seeks to develop a Critical Churn Risk Index (CCRI) that quantifies user vulnerability to churn. The findings will provide actionable insights for Spotify to refine its personalization strategies, optimize advertising placement, and design retention-focused interventions—ultimately enhancing user satisfaction and ensuring sustainable growth.')

st.divider()

st.dataframe(ds, use_container_width=True)

st.sidebar.image('user icon.png', caption = 'Welcome User')


# --- Configuration ---
MODEL_FILE_PATH = 'spotify_churn_model.joblib'

# Features your trained model actually expects
FEATURE_NAMES = [
    'age', 'gender', 'subscription_type', 'device_type', 'country',
    'listening_time', 'songs_played_per_day', 'skip_rate',
    'ads_listened_per_week', 'offline_listening'
    
]

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_FILE_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# --- Streamlit App ---
def main():
    st.title("🎵 Spotify Churn Prediction")
    st.markdown("Predict churn usin")

    # --- User Inputs ---
    st.header("User Profile & Activity")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider('Age', 15, 85, 30)
        gender = st.selectbox('Gender', ['Female', 'Male', 'Other'])
    
    with col2:
        country = st.selectbox('Country', ['CA', 'DE', 'AU', 'US', 'UK', 'IN', 'FR', 'PK'])
        subscription_type = st.selectbox('Subscription Type', ['Free', 'Family', 'Premium', 'Student'])
    
    with col3:
        device_type = st.selectbox('Device Type', ['Mobile', 'Desktop', 'Web'])
        offline_listening = 1 if st.checkbox('Offline Listening Enabled?', value=True) else 0

    st.header("Usage Metrics")
    col4, col5, col6, col7 = st.columns(4)
    
    with col4:
        listening_time = st.slider('Listening Time (minutes/day)', 0, 300, 150)
    with col5:
        songs_played_per_day = st.slider('Songs Played per Day', 0, 100, 50)
    with col6:
        skip_rate = st.number_input('Skip Rate (0.0-1.0)', 0.0, 1.0, 0.25, step=0.01)
    with col7:
        ads_listened_per_week = st.slider('Ads Listened (per week)', 0, 60, 5)

    # --- Prediction ---
    if st.button("Predict Churn"):
        input_dict = {
            'age': age,
            'gender': gender,
            'subscription_type': subscription_type,
            'device_type': device_type,
            'country': country,
            'listening_time': listening_time,
            'songs_played_per_day': songs_played_per_day,
            'skip_rate': skip_rate,
            'ads_listened_per_week': ads_listened_per_week,
            'offline_listening': offline_listening
        }

        input_df = pd.DataFrame([input_dict], columns=FEATURE_NAMES)

        with st.spinner("Calculating prediction..."):
            try:
                prediction = model.predict(input_df)[0]
                churn_proba = model.predict_proba(input_df)[:,1][0] if hasattr(model, "predict_proba") else None

                st.subheader("Prediction Result")
                if prediction == 1:
                    st.error("⚠️ High Churn Risk")
                else:
                    st.success("✅ Low Churn Risk")
                
                if churn_proba is not None:
                    st.info(f"Predicted probability of churn: {churn_proba*100:.2f}%")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
