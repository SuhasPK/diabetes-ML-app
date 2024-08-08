import streamlit as st # type: ignore
import pandas as pd # type: ignore

def run_eda_app():
    st.subheader('Exploratory Data Analysis')
    df = pd.read_csv('data/diabetes_data_upload.csv')
    st.dataframe(df)

