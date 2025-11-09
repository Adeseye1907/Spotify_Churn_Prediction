import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Configuration ---
MODEL_FILE_PATH = 'spotify_churn_model.joblib'

# Features your trained model actually expects
FEATURE_NAMES = [
    'age', 'gender', 'subscription_type', 'device_type', 'country',
    'listening_time', 'songs_played_per_day', 'skip_rate',
    'ads_listened_per_week', 'offline_listening'
    # Note: 'Coefficient' and 'CCRI' are NOT included unless you retrain
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
    st.title("🎵 Spotify Churn Prediction (Original Features)")
    st.markdown("Predict churn using the features your model was trained on.")

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
