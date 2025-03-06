import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('SWAN data.csv')
    
    # Select necessary columns
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
    
    # Feature Engineering
    df['CHRONIC_SCORE'] = df[['DIABETE10', 'HIGHBP10', 'HEART110', 'CHOLST110']].mean(axis=1)
    df['MENTAL_HEALTH_IDX'] = df[['DEPRESS10', 'SLEEPQL10', 'NERVES10']].mean(axis=1)
    df['HEALTH_SCORE'] = df[['CHRONIC_SCORE', 'MENTAL_HEALTH_IDX', 'OSTEOPO10', 'ANEMIA10', 'SLEEPQL10']].mean(axis=1)
    
    df['CHRONIC_SCORE'] = df['CHRONIC_SCORE'].fillna(df['CHRONIC_SCORE'].median())
    df['MENTAL_HEALTH_IDX'] = df['MENTAL_HEALTH_IDX'].fillna(df['MENTAL_HEALTH_IDX'].median())
    df['HEALTH_SCORE'] = df['HEALTH_SCORE'].fillna(df['HEALTH_SCORE'].median())
    
    selected_columns.extend(['CHRONIC_SCORE', 'MENTAL_HEALTH_IDX', 'HEALTH_SCORE'])
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    df = pd.DataFrame(df_scaled, columns=df.columns)
    
    return df, scaler, selected_columns

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
    
    df, scaler, selected_columns = load_data()
    model = load_model()
    
    # Sidebar Inputs
    st.sidebar.header("Patient Health Assessment")
    user_inputs = {col: 0 for col in selected_columns}  # Initialize with default values
    
    user_inputs.update({
        'AGE10': st.sidebar.slider("Age", 20, 80, 40),
        'DIABETE10': st.sidebar.selectbox("Diabetes", [0, 1]),
        'HIGHBP10': st.sidebar.selectbox("High Blood Pressure", [0, 1]),
        'DEPRESS10': st.sidebar.slider("Depression Score", 0.0, 5.0, 2.5),
        'SLEEPQL10': st.sidebar.slider("Sleep Quality (1=good, 5=poor)", 1, 5, 3),
        'EXERCIS10': st.sidebar.slider("Exercise Frequency (days/week)", 0, 7, 3)
    })
    
    # Create Input DataFrame
    input_data = pd.DataFrame([user_inputs])
    
    # Feature Engineering
    input_data['CHRONIC_SCORE'] = input_data[['DIABETE10', 'HIGHBP10', 'HEART110', 'CHOLST110']].mean(axis=1)
    input_data['MENTAL_HEALTH_IDX'] = input_data[['DEPRESS10', 'SLEEPQL10', 'NERVES10']].mean(axis=1)
    input_data['HEALTH_SCORE'] = input_data[['CHRONIC_SCORE', 'MENTAL_HEALTH_IDX', 'OSTEOPO10', 'ANEMIA10', 'SLEEPQL10']].mean(axis=1)
    
    if st.sidebar.button("Predict"):
        input_data = input_data[selected_columns]  # Ensure correct feature order
        input_data_scaled = scaler.transform(input_data)
        
        if input_data_scaled.shape[1] != model.n_features_in_:
            st.error(f"Feature mismatch! Model expects {model.n_features_in_} features but received {input_data_scaled.shape[1]}.")
            return
        
        cluster = model.predict(input_data_scaled)[0]
        
        st.subheader("Health Assessment Results")
        st.write(f"**Assigned Health Cluster:** {cluster}")

if __name__ == '__main__':
    main()
