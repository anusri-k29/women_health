import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.mixture import GaussianMixture

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('SWAN data.csv')
    # Select your columns
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
    # Fill any remaining missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    
    # Normalize select numerical columns
    scaler = MinMaxScaler()
    df[['AGE10', 'MDTALK10', 'NERVES10']] = scaler.fit_transform(df[['AGE10', 'MDTALK10', 'NERVES10']])
    
    # Label encode categorical columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        
    return df

# -------------------------------
# Load Pre-trained Model
# -------------------------------
@st.cache_data
def load_model():
    model = joblib.load('gmm_model.pkl')
    return model

# -------------------------------
# Health Classification Function
# -------------------------------
def classify_health(cluster_profile):
    risks = []
    if cluster_profile['CHRONIC_SCORE'] > 0.7:
        risks.append("High Cardiovascular Risk")
    elif cluster_profile['CHRONIC_SCORE'] > 0.5:
        risks.append("Moderate Cardiovascular Risk")
    if cluster_profile['MENTAL_HEALTH_IDX'] > 1.3:
        risks.append("Mental Health Support Needed")
    if cluster_profile['OSTEOPO10'] > 0.1:
        risks.append("Osteoporosis Risk")
    if cluster_profile['ANEMIA10'] > 1.2:
        risks.append("Potential Anemia")
    if cluster_profile['SLEEPQL10'] > 2.0:
        risks.append("Sleep Disorder Risk")
    return "Healthy" if not risks else ", ".join(risks)

# -------------------------------
# Main App Function
# -------------------------------
def main():
    st.title("Women's Health Cluster Analysis Dashboard")
    
    # Load data and model (ensure these files are in your repo)
    df = load_data()
    model = load_model()
    
    # Sidebar for patient input
    st.sidebar.header("Patient Health Assessment")
    age = st.sidebar.slider("Age", 20, 80, 40)
    diabetes = st.sidebar.selectbox("Diabetes", [0, 1])
    highbp = st.sidebar.selectbox("High Blood Pressure", [0, 1])
    depression = st.sidebar.slider("Depression Score", 0.0, 5.0, 2.5)
    sleep_quality = st.sidebar.slider("Sleep Quality (1=good, 5=poor)", 1, 5, 3)
    exercise = st.sidebar.slider("Exercise Frequency (days/week)", 0, 7, 3)
    
    # Create an input dataframe for prediction.
    # Here, we add default values for the rest of the features.
    input_data = pd.DataFrame({
        'AGE10': [age],
        'DIABETE10': [diabetes],
        'HIGHBP10': [highbp],
        'DEPRESS10': [depression],
        'SLEEPQL10': [sleep_quality],
        'EXERCIS10': [exercise],
        'PRGNAN10': [0],
        'LENGCYL10': [0],
        'ENDO10': [0],
        'MIGRAIN10': [0],
        'BROKEBO10': [0],
        'OSTEOPO10': [0],
        'HEART110': [0],
        'CHOLST110': [0],
        'THYROI110': [0],
        'INSULN110': [0],
        'NERVS110': [0],
        'ARTHRT110': [0],
        'FERTIL110': [0],
        'BCP110': [0],
        'REGVITA10': [0],
        'ONCEADA10': [0],
        'ANTIOXI10': [0],
        'VITCOMB10': [0],
        'VITAMNA10': [0],
        'BETACAR10': [0],
        'VITAMNC10': [0],
        'VITAMND10': [0],
        'VITAMNE10': [0],
        'CALCTUM10': [0],
        'IRON10': [0],
        'ZINC10': [0],
        'SELENIU10': [0],
        'FOLATE10': [0],
        'VTMSING10': [0],
        'YOGA10': [0],
        'DIETNUT10': [0],
        'SMOKERE10': [0],
        'MDTALK10': [0],
        'HLTHSER10': [0],
        'INSURAN10': [0],
        'NERVES10': [0],
        'RACE': [0]
    })
    
    # (Optional) Apply the same preprocessing steps to the input data if necessary.
    processed_input = input_data.copy()
    
    # When the user clicks "Predict", use the loaded model to predict the cluster.
    if st.sidebar.button("Predict"):
        cluster = model.predict(processed_input)[0]
        
        # Retrieve the cluster profile from your analysis.
        # (Assumes that your original data df has a 'Cluster' column from previous analysis.)
        if 'Cluster' in df.columns:
            cluster_profile = df[df['Cluster'] == cluster].mean()
        else:
            st.error("Cluster information missing from the data.")
            return
        
        st.subheader("Health Assessment Results")
        st.write(f"**Assigned Health Cluster:** {cluster}")
        
        health_status = classify_health(cluster_profile)
        st.write(f"**Health Status:** {health_status}")
        
        # Recommendations based on health status
        if "High Cardiovascular Risk" in health_status:
            st.warning("Recommended: Regular cardiac checkups and diet modification.")
        if "Mental Health Support Needed" in health_status:
            st.warning("Recommended: Counseling services and stress management.")

    # -------------------------------
    # Additional Data Visualizations
    # -------------------------------
    st.header("Population Health Insights")
    
    # Health Metric Distributions
    st.subheader("Health Metric Distributions")
    metric = st.selectbox("Select Metric", ['DIABETE10', 'HIGHBP10', 'DEPRESS10', 'SLEEPQL10'])
    fig = px.histogram(df, x=metric, color='Cluster', nbins=20)
    st.plotly_chart(fig)
    
    # Cluster Health Profiles
    st.subheader("Cluster Health Profiles")
    cluster_metrics = df.groupby('Cluster')[['DIABETE10', 'HIGHBP10', 'DEPRESS10', 'SLEEPQL10']].mean()
    st.bar_chart(cluster_metrics)
    
    # Feature Correlations
    st.subheader("Feature Correlations")
    corr = df.corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(corr, cmap='coolwarm')
    st.pyplot(plt)

if __name__ == '__main__':
    main()
