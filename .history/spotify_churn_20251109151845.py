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
st.markdown('Customer retention is a critical metric for digital streaming platforms such as Spotify, where user engagement and subscription continuity directly determine profitability. Despite offering a vast music library and personalized recommendations, Spotify continues to face challenges in minimizing user churn—the rate at which subscribers discontinue their service. Several behavioral and service-related factors influence churn tendencies. User engagement indicators such as listening time, skip rate, and offline listening frequency provide insight into satisfaction levels and platform loyalty. Likewise, exposure to advertisements and subscription type (free vs. premium) significantly shape user experience and long-term retention.

This study aims to examine the determinants of churn among Spotify users, with a focus on identifying patterns that predict disengagement. By analyzing listening behavior, ad interactions, and subscription preferences, the analysis seeks to develop a Critical Churn Risk Index (CCRI) that quantifies user vulnerability to churn. The findings will provide actionable insights for Spotify to refine its personalization strategies, optimize advertising placement, and design retention-focused interventions—ultimately enhancing user satisfaction and ensuring sustainable growth.')

st.divider()