import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
model = joblib.load('models/logistic_regression_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Function to predict diabetes risk
def predict_diabetes(features):
    # Convert features into a DataFrame
    features_df = pd.DataFrame([features], columns=[
        'age', 'gender', 'polyuria', 'polydipsia', 'sudden_weight_loss', 'weakness',
        'polyphagia', 'genital_thrush', 'visual_blurring', 'itching', 'irritability',
        'delayed_healing', 'partial_paresis', 'muscle_stiffness', 'alopecia', 'obesity'
    ])
    # Feature Scaling
    features_scaled = scaler.transform(features_df)
    # Predict probability
    probability = model.predict_proba(features_scaled)[0, 1]
    return probability

# Run the ML app
def run_ml_app():
    st.subheader("Machine Learning")
    
    st.write("### Input the features for diabetes risk prediction:")
    
    col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender = st.radio("Gender", options=["Female", "Male"], index=1)
        gender = 1 if gender == "Male" else 0
        polyuria = st.radio("Polyuria", options=["No", "Yes"], index=0)
        polyuria = 1 if polyuria == "Yes" else 0
        polydipsia = st.radio("Polydipsia", options=["No", "Yes"], index=0)
        polydipsia = 1 if polydipsia == "Yes" else 0
        sudden_weight_loss = st.radio("Sudden Weight Loss", options=["No", "Yes"], index=0)
        sudden_weight_loss = 1 if sudden_weight_loss == "Yes" else 0
    
    with col2:
        weakness = st.radio("Weakness", options=["No", "Yes"], index=0)
        weakness = 1 if weakness == "Yes" else 0
        polyphagia = st.radio("Polyphagia", options=["No", "Yes"], index=0)
        polyphagia = 1 if polyphagia == "Yes" else 0
        genital_thrush = st.radio("Genital Thrush", options=["No", "Yes"], index=0)
        genital_thrush = 1 if genital_thrush == "Yes" else 0
        visual_blurring = st.radio("Visual Blurring", options=["No", "Yes"], index=0)
        visual_blurring = 1 if visual_blurring == "Yes" else 0
    
    with col3:
        itching = st.radio("Itching", options=["No", "Yes"], index=0)
        itching = 1 if itching == "Yes" else 0
        irritability = st.radio("Irritability", options=["No", "Yes"], index=0)
        irritability = 1 if irritability == "Yes" else 0
        delayed_healing = st.radio("Delayed Healing", options=["No", "Yes"], index=0)
        delayed_healing = 1 if delayed_healing == "Yes" else 0
        partial_paresis = st.radio("Partial Paresis", options=["No", "Yes"], index=0)
        partial_paresis = 1 if partial_paresis == "Yes" else 0
    
    with col4:
        muscle_stiffness = st.radio("Muscle Stiffness", options=["No", "Yes"], index=0)
        muscle_stiffness = 1 if muscle_stiffness == "Yes" else 0
        alopecia = st.radio("Alopecia", options=["No", "Yes"], index=0)
        alopecia = 1 if alopecia == "Yes" else 0
        obesity = st.radio("Obesity", options=["No", "Yes"], index=0)
        obesity = 1 if obesity == "Yes" else 0
    
    features = [age, gender, polyuria, polydipsia, sudden_weight_loss, weakness,
                polyphagia, genital_thrush, visual_blurring, itching, irritability,
                delayed_healing, partial_paresis, muscle_stiffness, alopecia, obesity]
    
    if st.button("Predict"):
        probability = predict_diabetes(features)*100
        st.write(f"### Risk Probability: {probability:.2f}")
        
        if probability > 0.5:
            st.warning("#### Based on the provided features, the risk of diabetes is high.")
        else:
            st.success("#### Based on the provided features, the risk of diabetes is low.")
