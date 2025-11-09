import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# ---------- Load dataset (for display and optional z_score calculation) ----------
@st.cache_data
def load_dataset(path='spotify_churn_dataset.csv'):
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Could not load dataset at '{path}': {e}")
        return None

ds = load_dataset('spotify_churn_dataset.csv')

# ---------- Page header & visuals ----------
st.markdown("<h1 style='color: #DD5746; text-align: center; font-size:60px; font-family: Monospace'>SPOTIFY CHURN PREDICTION</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='margin:10px; color:#FFC470; text-align:center; font-family:Serif'>Built by Adeseye</h4>", unsafe_allow_html=True)
st.image('spotify.png')
st.divider()

st.markdown("<h2 style='color: #F7C566; text-align: center; font-family: montserrat'>Background Of Study</h2>", unsafe_allow_html=True)
st.markdown(
    "Customer retention is a critical metric for digital streaming platforms such as Spotify... "
    "This study aims to examine the determinants of churn among Spotify users, with a focus on identifying patterns that predict disengagement."
)

st.divider()

if ds is not None:
    st.dataframe(ds, use_container_width=True)

st.sidebar.image('user icon.png', caption='Welcome User')

# ---------- Configuration ----------
MODEL_FILE_PATH = 'spotify_churn_model.joblib'

# FEATURE_NAMES must match EXACTLY the columns used during training (order matters for some uses)
FEATURE_NAMES = [
 'gender' 'age' 'country' 'subscription_type' 'listening_time'
 'songs_played_per_day' 'skip_rate' 'device_type' 'ads_listened_per_week'
 'offline_listening']
# ---------- Load model ----------
@st.cache_resource
def load_model(path=MODEL_FILE_PATH):
    try:
        m = joblib.load(path)
        return m
    except FileNotFoundError:
        st.sidebar.error(f"Model file not found at '{path}'.")
        return None
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return None

model = load_model()

# ---------- Attempt to compute AUC (best-effort) ----------
def compute_auc_if_possible(model, ds, feature_names, label_col='is_churned'):
    """Try to compute AUC using dataset and model. Return float or None and message."""
    if model is None or ds is None:
        return None, "Model or dataset not loaded."
    # Check if label exists
    if label_col not in ds.columns:
        return None, f"Label column '{label_col}' not found in dataset."
    # Check that all features exist in ds
    missing = [c for c in feature_names if c not in ds.columns]
    if missing:
        return None, f"Missing features in dataset for AUC computation: {missing}"
    # Build X and y
    try:
        X = ds[feature_names].copy()
        y = ds[label_col]
        # If model needs numeric types for some categorical columns that were strings at fit time,
        # the predict_proba call may fail. We'll try and catch exceptions.
        if not hasattr(model, "predict_proba"):
            return None, "Model has no predict_proba method; cannot compute AUC."
        probs = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probs)
        return float(auc), "AUC computed on provided dataset."
    except Exception as e:
        return None, f"AUC computation failed: {e}"

AUC_SCORE = None
AUC_MSG = "AUC not computed."

if model is not None and ds is not None:
    AUC_SCORE, AUC_MSG = compute_auc_if_possible(model, ds, FEATURE_NAMES)
# Sidebar display: show computed AUC or placeholder control to manually enter a value
st.sidebar.header('Model Performance')
if AUC_SCORE is not None:
    st.sidebar.metric(label='AUC ROC Score (on dataset)', value=f'{AUC_SCORE:.3f}')
    if AUC_SCORE >= 0.9:
        st.sidebar.success("Excellent model discrimination.")
    elif AUC_SCORE >= 0.8:
        st.sidebar.info("Good model discrimination.")
    else:
        st.sidebar.warning("Model discrimination may be weak.")
else:
    st.sidebar.info(AUC_MSG)
    # Allow user to enter an AUC manually if they want to show a known score
    manual_auc = st.sidebar.number_input('Enter known AUC (optional):', min_value=0.0, max_value=1.0, step=0.01, value=0.0)
    if manual_auc > 0:
        st.sidebar.metric(label='AUC ROC (manual)', value=f'{manual_auc:.3f}')

st.markdown("---")

# ---------- Main App UI ----------
def main():
    st.title('🎵 Spotify Churn Prediction App (Fixed)')
    st.markdown("Predict churn using the features your model was trained on. Provide `user_id` and `z_score` if available; otherwise z_score will be approximated from dataset statistics.")

    # Innovation inputs (kept for display; not used by model unless model was trained on them)
    st.header('🔬 Optional Innovation Inputs (not used unless model was trained with them)')
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        coefficient = st.number_input('Coefficient (optional)', min_value=-10.0, max_value=10.0, value=0.5, step=0.01)
    with col_i2:
        ccri = st.number_input('CCRI (optional)', min_value=0.0, max_value=100.0, value=30.0, step=0.1)

    st.markdown('---')
    st.header('User Profile and Activity')

    # Demographics
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider('Age', min_value=15, max_value=120, value=30)  # allow beyond 30
        gender = st.selectbox('Gender', options=['Female', 'Male', 'Other'])
    with col2:
        country = st.selectbox('Country', options=['AU', 'CA', 'DE', 'FR', 'IN', 'PK', 'UK', 'US'])
        subscription_type = st.selectbox('Subscription Type', options=['Free', 'Family', 'Premium', 'Student'])
    with col3:
        device_type = st.selectbox('Device Type', options=['Desktop', 'Web', 'Mobile'])
        offline_listening_flag = st.checkbox('Enables Offline Listening?', value=True)
        offline_listening = 1 if offline_listening_flag else 0

    st.markdown('---')
    st.header('Usage Metrics')
    col4, col5, col6, col7 = st.columns(4)
    with col4:
        listening_time = st.slider('Listening Time (minutes/day)', min_value=0, max_value=1440, value=150)
    with col5:
        songs_played_per_day = st.slider('Songs Played Per Day', min_value=0, max_value=1000, value=50)
    with col6:
        skip_rate = st.number_input('Skip Rate (0.0 to 1.0)', min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    with col7:
        ads_listened_per_week = st.slider('Ads Listened (per week)', min_value=0, max_value=500, value=5)

    st.markdown('---')
    st.header('Identifiers and derived features')

    col8, col9 = st.columns(2)
    with col8:
        user_id = st.text_input('User ID (optional)', value='')  # optional; can be left blank
    with col9:
        zscore_input = st.text_input('z_score (optional - enter if you have it)', value='')

    # Predict button
    if st.button('Predict Churn Likelihood'):
        if model is None:
            st.error("Model not loaded. Check MODEL_FILE_PATH and model format.")
            return

        # Prepare z_score: use user-provided if present; else attempt to compute from dataset using listening_time
        if zscore_input.strip() != '':
            try:
                z_score = float(zscore_input)
            except:
                st.warning("Could not parse provided z_score; defaulting to 0.0.")
                z_score = 0.0
        else:
            # Best-effort automatic z_score derivation:
            # If dataset has 'listening_time' column, standardize current listening_time using dataset mean/std
            if ds is not None and 'listening_time' in ds.columns:
                mean_lt = ds['listening_time'].mean()
                std_lt = ds['listening_time'].std(ddof=0)
                if std_lt == 0 or np.isnan(std_lt):
                    z_score = 0.0
                else:
                    z_score = (listening_time - mean_lt) / std_lt
            else:
                # Fallback default
                z_score = 0.0

        # user_id fallback: if empty, set numeric 0 or keep string depending on training
        if user_id.strip() == '':
            # If your model used numeric user_id, set 0; otherwise set 'unknown'
            user_id_val = 0
        else:
            # try numeric cast, else keep string
            try:
                user_id_val = int(user_id)
            except:
                user_id_val = user_id

        # Build input data in the exact FEATURE_NAMES order
        # If your model was trained with raw categorical columns (strings), we pass strings.
        input_data = {
            'user_id': user_id_val,
            'z_score': z_score,
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

        # Create DataFrame respecting the feature order
        try:
            input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        except Exception as e:
            st.error(f"Failed to construct input DataFrame with the required FEATURE_NAMES: {e}")
            st.write("FEATURE_NAMES expected:", FEATURE_NAMES)
            st.write("Input keys:", list(input_data.keys()))
            return

        # Perform prediction
        with st.spinner('Calculating Churn Prediction...'):
            try:
                # If model requires different dtypes/encoding, ensure you apply same preprocessing as in training.
                prediction = model.predict(input_df)[0]
                churn_proba = model.predict_proba(input_df)[:, 1][0] if hasattr(model, 'predict_proba') else None

                st.subheader('Prediction Result')
                if prediction == 1:
                    st.error('⚠️ High Churn Risk Predicted')
                else:
                    st.success('✅ Low Churn Risk Predicted')

                if churn_proba is not None:
                    st.info(f"The predicted probability of **Churn** is **{churn_proba * 100:.2f}%**.")
                else:
                    st.info("Model does not expose predict_proba; only class prediction shown.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.write("Tip: Ensure the model was trained on the exact same feature names, types and order as provided here.")
                # helpful debug info
                try:
                    st.write("Model expects:", model.feature_names_in_ if hasattr(model, 'feature_names_in_') else "unknown")
                except:
                    pass

if __name__ == '__main__':
    main()
