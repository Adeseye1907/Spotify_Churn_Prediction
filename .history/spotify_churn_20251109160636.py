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


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Configuration ---
MODEL_FILE_PATH = 'spotify_churn_model.joblib' 
# CRITICAL: This list now includes 23 features, which we assume were used during training.
# The order is paramount: it must match the column order in your trained model's data.
FEATURE_NAMES = [
    'user'
    'Coefficient',                  # 1. Your Innovation 1
    'CCRI',                         # 2. Your Innovation 2
    'age',                          # 3.
    'listening_time',               # 4.
    'songs_played_per_day',         # 5.
    'skip_rate',                    # 6.
    'ads_listened_per_week',        # 7.
    'offline_listening',            # 8.
    
    # DUMMIES (15 features total, implying 'Male', 'Free', and 'Desktop' were dropped)
    'gender_Female',                # 9.
    'gender_Other',                 # 10.
    
    'subscription_type_Family',     # 11.
    'subscription_type_Premium',    # 12.
    'subscription_type_Student',    # 13.
    
    'device_type_Mobile',           # 14.
    'device_type_Web',              # 15.
    
    # Country Dummies (Implies NO country was dropped)
    'country_AU',                   # 16.
    'country_CA',                   # 17.
    'country_DE',                   # 18.
    'country_FR',                   # 19.
    'country_IN',                   # 20.
    'country_PK',                   # 21.
    'country_UK',                   # 22.
    'country_US',                   # 23.
]


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
    st.title('🎵 Spotify Churn Prediction App with Innovation')
    st.markdown("""
        Predict the likelihood of a user churning based on activity, demographics, and your innovative **Coefficient** and **CCRI** features.
    """)
    
    # ----------------------------------------------------------------------
    # 🌟 INNOVATION INPUTS (Coefficient and CCRI)
    # ----------------------------------------------------------------------
    st.header('🔬 Innovative Input Features')
    col_i1, col_i2 = st.columns(2)
    
    with col_i1:
        coefficient = st.number_input(
            'Coefficient', 
            min_value=-10.0, max_value=10.0, value=0.5, step=0.01, 
            help="Your custom calculated feature value."
        )
    
    with col_i2:
        ccri = st.number_input(
            'CCRI (Customer Churn Risk Index)', 
            min_value=0.0, max_value=100.0, value=30.0, step=0.1, 
            help="Your custom calculated risk index."
        )

    st.markdown('---')
    
    # ----------------------------------------------------------------------
    # SPOTIFY DATASET INPUTS
    # ----------------------------------------------------------------------
    st.header('User Profile and Activity')
    
    # Row 1: Demographics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider('Age', min_value=15, max_value=85, value=30)
        gender = st.selectbox('Gender', options=['Female', 'Male', 'Other'])
    
    with col2:
        country = st.selectbox('Country', options=['AU', 'CA', 'DE', 'FR', 'IN', 'PK', 'UK', 'US'])
        subscription_type = st.selectbox(
            'Subscription Type', 
            options=['Free', 'Family', 'Premium', 'Student']
        )
    
    with col3:
        device_type = st.selectbox('Device Type', options=['Desktop', 'Web', 'Mobile'])
        offline_listening_flag = st.checkbox('Enables Offline Listening?', value=True)
        # Convert boolean to 0/1 integer
        offline_listening = 1 if offline_listening_flag else 0
        
    st.markdown('---')
    st.header('Usage Metrics')
    
    # Row 2: Usage
    col4, col5, col6, col7 = st.columns(4)
    
    with col4:
        listening_time = st.slider(
            'Listening Time (minutes/day)', 
            min_value=0, max_value=300, value=150
        )
    
    with col5:
        songs_played_per_day = st.slider(
            'Songs Played Per Day', 
            min_value=0, max_value=100, value=50
        )
    
    with col6:
        skip_rate = st.number_input(
            'Skip Rate (0.0 to 1.0)', 
            min_value=0.0, max_value=1.0, value=0.25, step=0.01
        )
        
    with col7:
        ads_listened_per_week = st.slider(
            'Ads Listened (per week)', 
            min_value=0, max_value=60, value=5
        )

    
    if st.button('Predict Churn Likelihood'):
        
        # --- Data Preprocessing (CRITICAL STEP) ---
        # 1. Create a dictionary of raw inputs
        input_dict_raw = {
            'gender': [gender],
            'subscription_type': [subscription_type],
            'device_type': [device_type],
            'country': [country]
        }
        encoded_df = pd.DataFrame(input_dict_raw)
        
        # 2. Perform One-Hot Encoding to get the 18 dummy features
        # We perform OHE on all categories first, and then manually select the required 15 dummies.
        encoded_df = pd.get_dummies(encoded_df, prefix=['gender', 'subscription_type', 'device_type', 'country'])
        
        # 3. Initialize the final feature dictionary (all 23 expected features set to 0 initially)
        final_features = {name: 0 for name in FEATURE_NAMES}

        # Add INNOVATION and original numerical/binary features
        final_features.update({
            'Coefficient': coefficient,
            'CCRI': ccri,
            'age': age,
            'listening_time': listening_time,
            'songs_played_per_day': songs_played_per_day,
            'skip_rate': skip_rate,
            'ads_listened_per_week': ads_listened_per_week,
            'offline_listening': offline_listening
        })

        # 4. Map the 15 required OHE features from the encoded DataFrame
        for col in final_features.keys():
            # Check if the feature name is an OHE column AND if it exists in the encoded input
            if col in encoded_df.columns and encoded_df[col].iloc[0] == True:
                final_features[col] = 1
        
        # 5. Create the final DataFrame for prediction, ensuring correct order
        input_array = np.array([final_features[name] for name in FEATURE_NAMES]).reshape(1, -1)
        input_df = pd.DataFrame(input_array, columns=FEATURE_NAMES)
        
        # 6. Make the Prediction
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
                st.error(f"A new error occurred during prediction: {e}")

if __name__ == '__main__':
    main()
