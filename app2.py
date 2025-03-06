import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('SWAN data.csv')
    
    # Use original 44 features that match the trained model
    selected_columns = [
        'AGE10', 'PRGNAN10', 'LENGCYL10', 'ENDO10', 'ANEMIA10', 'DIABETE10', 
        'HIGHBP10', 'MIGRAIN10', 'BROKEBO10', 'OSTEOPO10', 'HEART110', 'CHOLST110',
        'THYROI110', 'INSULN110', 'NERVS110', 'ARTHRT110', 'FERTIL110', 'BCP110',
        'REGVITA10', 'ONCEADA10', 'ANTIOXI10', 'VITCOMB10', 'VITAMNA10', 'BETACAR10',
        'VITAMNC10', 'VITAMND10', 'VITAMNE10', 'CALCTUM10', 'IRON10', 'ZINC10',
        'SELENIU10', 'FOLATE10', 'VTMSING10', 'EXERCIS10', 'YOGA10', 'DIETNUT10',
        'SMOKERE10', 'MDTALK10', 'HLTHSER10', 'INSURAN10', 'NERVES10', 'DEPRESS10',
        'SLEEPQL10', 'RACE'
    ]
    df = df[selected_columns]
    
    # Handle missing values
    categorical_cols = ["RACE", "INSURAN10", "HLTHSER10", "SMOKERE10"]
    numerical_cols = ["DEPRESS10", "SLEEPQL10", "PRGNAN10", "DIABETE10", "HIGHBP10", "MDTALK10"]
    
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Encode categorical features
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    return pd.DataFrame(df_scaled, columns=df.columns), scaler

# -------------------------------
# Load Pre-trained Model
# -------------------------------
@st.cache_data
def load_model():
    return joblib.load('gmm_model.pkl')

# -------------------------------
# Streamlit UI
# -------------------------------
def main():
    st.title("Women's Health Cluster Analysis Dashboard")
    
    df, scaler = load_data()
    model = load_model()
    
    st.header("Patient Health Assessment")
    
    user_inputs = {col: 0.0 for col in df.columns}  # Initialize all features
    
    # Collect user inputs directly on the main page
    user_inputs.update({
        'AGE10': st.slider("Age", 20, 80, 40),
        'DIABETE10': st.selectbox("Diabetes", [0, 1]),
        'HIGHBP10': st.selectbox("High Blood Pressure", [0, 1]),
        'DEPRESS10': st.slider("Depression Score", 0.0, 5.0, 2.5),
        'SLEEPQL10': st.slider("Sleep Quality (1=good, 5=poor)", 1, 5, 3),
        'EXERCIS10': st.slider("Exercise Frequency (days/week)", 0, 7, 3),
        'MIGRAIN10': st.selectbox("Migraine", [0, 1]),
        'OSTEOPO10': st.selectbox("Osteoporosis", [0, 1]),
        'THYROI110': st.selectbox("Thyroid Disorder", [0, 1]),
        'NERVS110': st.selectbox("Nerve Problems", [0, 1]),
        'ARTHRT110': st.selectbox("Arthritis", [0, 1]),
        'SMOKERE10': st.selectbox("Smoking Status", [0, 1]),
        'YOGA10': st.slider("Yoga Practice (days/week)", 0, 7, 2),
        'DIETNUT10': st.slider("Diet Quality (1=poor, 5=excellent)", 1, 5, 3)
    })
    
    if st.button("Predict"):
        # Create input DataFrame with ALL features
        input_data = pd.DataFrame([user_inputs])[df.columns]  # Ensure correct order
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Predict cluster
        cluster = model.predict(input_scaled)[0]
        
        st.subheader("Health Assessment Results")
        st.write(f"**Assigned Health Cluster:** {cluster}")
        
        # Cluster Interpretation
        cluster_info = {
            0: "🚨 **High-Risk Profile**: This cluster indicates individuals with significant health risks, such as chronic conditions (e.g., diabetes, high blood pressure) and poor mental health. Immediate lifestyle changes and medical interventions are recommended.",
            1: "⚠️ **Moderate-Risk Profile**: This cluster represents individuals with manageable health conditions. Regular check-ups, moderate exercise, and a balanced diet are advised to prevent further complications.",
            2: "✅ **Low-Risk Profile**: This cluster includes individuals with good overall health. Maintaining healthy habits, such as regular exercise and a nutritious diet, is recommended to sustain this positive health status."
        }
        
        st.write(cluster_info.get(cluster, "Unknown Cluster"))

if __name__ == '__main__':
    main()
