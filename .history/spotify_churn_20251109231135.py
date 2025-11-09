this is what i have, and i want the code for the features that was trained and then for the CCRI# --- Configuration ---
MODEL_FILE_PATH = 'spotify_churn_model.joblib' 
# IMPORTANT: The list of feature names your model expects, in the EXACT order.
# This must match the training order: [Coefficient, CCRI, original features...]
FEATURE_NAMES = [
    'Coefficient',                  # Your Innovation 1
    'CCRI',                         # Your Innovation 2
    'age', 'listening_time', 'songs_played_per_day', 'skip_rate', 
    'ads_listened_per_week', 'offline_listening',
    
    # One-Hot Encoded Gender (assuming Male is dropped as reference category)
    'gender_Female', 'gender_Other', 
    
    # One-Hot Encoded Subscription Type (assuming Free is dropped)
    'subscription_type_Family', 'subscription_type_Premium', 'subscription_type_Student',
    
    # One-Hot Encoded Device Type (assuming Desktop is dropped)
    'device_type_Mobile', 'device_type_Web',
    
    # Country features (You must ensure this list matches the columns created by your model's encoding)
    'country_CA', 'country_DE', 'country_AU', 'country_US', 'country_UK', 'country_IN', 
    'country_FR', 'country_PK' 
]


# --- Load Model with Caching for Speed ---
@st.cache_resource
def load_model():
    """Loads the trained model from the file."""
    try:
        model = joblib.load(MODEL_FILE_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at '{MODEL_FILE_PATH}'.")
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
    
    # --- INNOVATION INPUTS ---
    st.header('🔬 Innovative Input Features')
    col_i1, col_i2 = st.columns(2)
    
    with col_i1:
        # User input for your proprietary 'Coefficient'
        coefficient = st.number_input(
            'Coefficient (e.g., from -5.0 to 5.0)', 
            min_value=-10.0, max_value=10.0, value=0.5, step=0.01, 
            help="Your first innovative feature value."
        )
    
    with col_i2:
        # User input for your proprietary 'CCRI'
        ccri = st.number_input(
            'CCRI (Customer Churn Risk Index, e.g., 0 to 100)', 
            min_value=0.0, max_value=100.0, value=30.0, step=0.1, 
            help="Your second innovative feature value."
        )

    st.markdown('---')
    
    # --- SPOTIFY DATASET INPUTS ---
    st.header('User Profile and Activity')
    
    # Row 1: Demographics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider('Age', min_value=15, max_value=85, value=30)
        gender = st.selectbox('Gender', options=['Female', 'Male', 'Other'])
    
    with col2:
        country = st.selectbox('Country', options=['CA', 'DE', 'AU', 'US', 'UK', 'IN', 'FR', 'PK'])
        subscription_type = st.selectbox(
            'Subscription Type', 
            options=['Free', 'Family', 'Premium', 'Student']
        )
    
    with col3:
        device_type = st.selectbox('Device Type', options=['Mobile', 'Desktop', 'Web'])
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
            min_value=0, max_value=60, value=5, help="Only relevant for Free/Student tiers."
        )

    
    if st.button('Predict Churn Likelihood'):
        
        # --- Data Preprocessing (CRITICAL STEP) ---
        # 1. Create a dummy dataframe for encoding categorical variables
        input_dict_cat = {
            'gender': [gender],
            'subscription_type': [subscription_type],
            'device_type': [device_type],
            'country': [country]
        }
        encoded_df = pd.DataFrame(input_dict_cat)
        
        # 2. Perform One-Hot Encoding
        encoded_df = pd.get_dummies(encoded_df, drop_first=False) 
        
        # 3. Create the final feature vector
        # Initialize a dictionary with all expected features set to 0
        final_features = {name: 0 for name in FEATURE_NAMES}

        # Add INNOVATION and numerical/binary features
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

        # Add the encoded categorical features (handle columns dropped by model training)
        for col in encoded_df.columns:
            # Check if the encoded column is one that was used during training
            if col in final_features and encoded_df[col].iloc[0] == True:
                final_features[col] = 1
        
        # 4. Create the final DataFrame for prediction, ensuring correct order
        input_array = np.array([final_features[name] for name in FEATURE_NAMES]).reshape(1, -1)
        input_df = pd.DataFrame(input_array, columns=FEATURE_NAMES)
        
        # 5. Make the Prediction
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
                st.error(f"An error occurred during prediction. Please ensure your model inputs match the required **{len(FEATURE_NAMES)}** features: {e}")

if __name__ == '__main__':
    main()