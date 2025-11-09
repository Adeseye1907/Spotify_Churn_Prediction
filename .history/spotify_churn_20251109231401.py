import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Configuration ---
MODEL_FILE_PATH = 'spotify_churn_model.joblib' 

# CRITICAL: 12 Feature Names in the EXACT order your trained model expects.
FEATURE_NAMES = [
    'user_id',
    'gender',
    'age',
    'country',
    'subscription_type',
    'listening_time',
    'songs_played_per_day',
    'skip_rate',
    'device_type',
    'ads_listened_per_week',
    'offline_listening',
    'z_score'
]

# --- STATIC MODEL METRIC ---
# IMPORTANT: Replace 0.85 with your model's actual AUC ROC score from the test set.
STATIC_AUC_SCORE = 0.85 


# --- Load Model with Caching for Speed ---
@st.cache_resource
def load_model():
    """Loads the trained model from the file."""
    try:
        model = joblib.load(MODEL_FILE_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at '{MODEL_FILE_PATH}'. Please check the file name.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

model = load_model()

# --- Streamlit UI and Prediction Logic ---
def main():
    st.title('🎵 Spotify Churn Prediction App')
    
    # ----------------------------------------------------------------------
    # 📊 MODEL PERFORMANCE METRIC (Sidebar)
    # ----------------------------------------------------------------------
    st.sidebar.header('Model Performance')
    st.sidebar.metric(
        label='AUC ROC Score (on Test Set)', 
        value=f'{STATIC_AUC_SCORE:.3f}',
        delta='Higher is better'
    )
    if STATIC_AUC_SCORE >= 0.8:
        st.sidebar.success("Model is highly discriminative.")
    elif STATIC_AUC_SCORE >= 0.7:
        st.sidebar.info("Model performance is acceptable.")
    else:
        st.sidebar.warning("Model performance may be poor.")
    
    st.markdown("""
        Enter the user's attributes below to predict churn likelihood. 
        Ensure you use values that characterize high-risk users (e.g., high skip rate, low listening time) 
        to test the model's ability to predict churn.
    """)
    st.markdown("---")
    
    # ----------------------------------------------------------------------
    # PROFILE AND DEMOGRAPHICS
    # ----------------------------------------------------------------------
    st.header('User Profile')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 1. user_id
        user_id = st.number_input('User ID', min_value=1, value=8001, step=1)
        # 2. age
        age = st.slider('Age', min_value=15, max_value=85, value=30)
    
    with col2:
        # 3. gender
        gender = st.selectbox('Gender', options=['Female', 'Male', 'Other'])
        # 4. country
        country = st.selectbox('Country', options=['CA', 'DE', 'AU', 'US', 'UK', 'IN', 'FR', 'PK'])
    
    with col3:
        # 5. subscription_type
        subscription_type = st.selectbox(
            'Subscription Type', 
            options=['Free', 'Family', 'Premium', 'Student']
        )
        # 6. device_type
        device_type = st.selectbox('Device Type', options=['Desktop', 'Web', 'Mobile'])
        # 7. offline_listening (Input for the binary feature)
        offline_listening = 1 if st.checkbox('Enables Offline Listening?', value=True) else 0

    st.markdown('---')
    
    # ----------------------------------------------------------------------
    # USAGE METRICS AND Z-SCORE (Engineered Feature)
    # ----------------------------------------------------------------------
    st.header('Usage Metrics')
    col4, col5, col6, col7 = st.columns(4)
    
    with col4:
        # 8. listening_time
        listening_time = st.slider(
            'Listening Time (minutes/day)', 
            min_value=0, max_value=300, value=150
        )
    
    with col5:
        # 9. songs_played_per_day
        songs_played_per_day = st.slider(
            'Songs Played Per Day', 
            min_value=0, max_value=100, value=50
        )
    
    with col6:
        # 10. skip_rate
        skip_rate = st.number_input(
            'Skip Rate (0.0 to 1.0)', 
            min_value=0.0, max_value=1.0, value=0.25, step=0.01
        )
        
    with col7:
        # 11. ads_listened_per_week
        ads_listened_per_week = st.slider(
            'Ads Listened (per week)', 
            min_value=0, max_value=60, value=5
        )
    
    st.subheader('Engineered Feature')
    # 12. z_score (Your key innovation input)
    z_score = st.number_input(
        'Z-Score (Engineered Value)', 
        min_value=-10.0, max_value=10.0, value=0.0, step=0.01,
        help="This must be the pre-calculated value expected by the model (e.g., your CCRI or Coefficient)."
    )

    
    if st.button('Predict Churn Likelihood'):
        
        # 1. Create a dictionary of all 12 feature values
        input_data = {
            'user_id': user_id,
            'gender': gender,
            'age': age,
            'country': country,
            'subscription_type': subscription_type,
            'listening_time': listening_time,
            'songs_played_per_day': songs_played_per_day,
            'skip_rate': skip_rate,
            'device_type': device_type,
            'ads_listened_per_week': ads_listened_per_week,
            'offline_listening': offline_listening,
            'z_score': z_score
        }

        # 2. Convert to DataFrame and ensure the feature order is EXACTLY correct
        # This step is critical for matching the model's expected column order.
        input_df = pd.DataFrame([input_data])[FEATURE_NAMES]
        
        # 3. Make the Prediction
        with st.spinner('Calculating Churn Prediction...'):
            try:
                # Prediction (0 or 1)
                prediction = model.predict(input_df)[0]
                
                # Prediction probability (likelihood of being the "1" class - Churn)
                churn_proba = model.predict_proba(input_df)[:, 1][0] if hasattr(model, 'predict_proba') else None
                
                # --- Display Result ---
                st.subheader('Prediction Result')
                
                if prediction == 1:
                    st.error('⚠️ High Churn Risk Predicted')
                else:
                    st.success('✅ Low Churn Risk Predicted')
                
                if churn_proba is not None:
                    percent_proba = churn_proba * 100
                    st.info(f"The predicted probability of **Churn** is **{percent_proba:.2f}%**.")
                    
            except Exception as e:
                st.error(f"Prediction Error. The model failed to predict. Please ensure your model (churn_model.joblib) supports string categorical features: {e}")

if __name__ == '__main__':
    main()