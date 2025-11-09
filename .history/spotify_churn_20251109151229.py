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
