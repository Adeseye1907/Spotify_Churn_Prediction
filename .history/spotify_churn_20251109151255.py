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
st.markdown('Advertising plays a pivotal role in shaping consumer awareness and driving sales. Businesses allocate significant resources to advertising, aiming to reach target audiences and communicate value propositions effectively. However, quantifying the direct impact of advertising on sales performance remains a challenge, with debates surrounding its cost-effectiveness and long-term benefits. Understanding the relationship between advertising efforts and sales outcomes is essential for optimizing marketing strategies and ensuring a positive return on investment (ROI). This study seeks to examine how advertising influences consumer purchasing behavior and contributes to sales growth, providing actionable insights for businesses to maximize their marketing efficiency.')

st.divider()