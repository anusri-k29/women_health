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

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    df = pd.DataFrame(df_scaled, columns=df.columns)

    return df, scaler  # Return scaler as well

# -------------------------------
# Load Pre-trained Model
# -------------------------------
@st.cache_data
def load_model():
    return joblib.load('gmm_model.pkl')

# -------------------------------
# Health Classification
# -------------------------------
def classify_health(cluster_profile):
    risks = []

    if cluster_profile['CHRONIC_SCORE'] > 0.8:
        risks.append("🚨 Critical Cardiac Risk")
    elif cluster_profile['CHRONIC_SCORE'] > 0.6:
        risks.append("⚠️ Moderate Cardiovascular Risk")

    if cluster_profile['MENTAL_HEALTH_IDX'] > 1.5:
        risks.append("🧠 Urgent Mental Health Needs")
    elif cluster_profile['MENTAL_HEALTH_IDX'] > 1.2:
        risks.append("😟 Mild Mental Health Support Needed")

    if cluster_profile['OSTEOPO10'] > 0.15:
        risks.append("🦴 High Osteoporosis Risk")
    elif cluster_profile['OSTEOPO10'] > 0.05:
        risks.append("🦴 Moderate Bone Health Risk")

    if cluster_profile['ANEMIA10'] > 1.3:
        risks.append("🩸 Severe Anemia Risk")
    elif cluster_profile['ANEMIA10'] > 1.1:
        risks.append("🩸 Mild Anemia Signs")

    if cluster_profile['SLEEPQL10'] > 2.2:
        risks.append("💤 Critical Sleep Issues")
    elif cluster_profile['SLEEPQL10'] > 1.8:
        risks.append("💤 Mild Sleep Disturbances")

    if cluster_profile['HEALTH_SCORE'] < 0.9 and len(risks) < 2:
        return "✅ Healthy Profile"
    elif not risks:
        return "🌟 Generally Healthy"
    else:
        return " | ".join(risks)

# -------------------------------
# Streamlit UI
# -------------------------------
def main():
    st.title("Women's Health Cluster Analysis Dashboard")

    df, scaler = load_data()
    model = load_model()

    # Sidebar Inputs
    st.sidebar.header("Patient Health Assessment")
    age = st.sidebar.slider("Age", 20, 80, 40)
    diabetes = st.sidebar.selectbox("Diabetes", [0, 1])
    highbp = st.sidebar.selectbox("High Blood Pressure", [0, 1])
    depression = st.sidebar.slider("Depression Score", 0.0, 5.0, 2.5)
    sleep_quality = st.sidebar.slider("Sleep Quality (1=good, 5=poor)", 1, 5, 3)
    exercise = st.sidebar.slider("Exercise Frequency (days/week)", 0, 7, 3)

    # Create Input DataFrame
    input_data = pd.DataFrame({
        'AGE10': [age],
        'DIABETE10': [diabetes],
        'HIGHBP10': [highbp],
        'DEPRESS10': [depression],
        'SLEEPQL10': [sleep_quality],
        'EXERCIS10': [exercise],
        'OSTEOPO10': [0], 'ANEMIA10': [0], 'NERVES10': [0], 'HEART110': [0], 'CHOLST110': [0]  # Default values
    })

    # Feature Engineering
    input_data['CHRONIC_SCORE'] = input_data[['DIABETE10', 'HIGHBP10', 'HEART110', 'CHOLST110']].mean(axis=1)
    input_data['MENTAL_HEALTH_IDX'] = input_data[['DEPRESS10', 'SLEEPQL10', 'NERVES10']].mean(axis=1)
    input_data['HEALTH_SCORE'] = input_data[['CHRONIC_SCORE', 'MENTAL_HEALTH_IDX', 'OSTEOPO10', 'ANEMIA10', 'SLEEPQL10']].mean(axis=1)

    if st.sidebar.button("Predict"):
    # Ensure input_data matches the scaler's expected features
        input_data = pd.DataFrame([input_data.to_dict(orient='records')[0]], columns=scaler.feature_names_in_)
    
        # Debugging prints
        print("Scaler expected columns:", scaler.feature_names_in_)
        print("Input columns:", input_data.columns)
        
        # Convert to numeric and fill NaNs
        input_data = input_data.apply(pd.to_numeric, errors='coerce').fillna(0)
    
        # Transform input using the scaler
        processed_input = scaler.transform(input_data)
    
        # Predict Cluster
        # Ensure processed_input is in the correct shape
        print(f"Original Processed Input Shape: {np.shape(processed_input)}")
        
        if hasattr(model, "n_features_in_"):
            expected_features = model.n_features_in_
            print(f"Model expects {expected_features} features")
        print(f"Processed Input Shape Before Reshaping: {np.shape(processed_input)}")
        
        # Reshape input to match the expected format
        processed_input = np.array(processed_input)
        
        print(f"Processed Input Shape After Conversion: {processed_input.shape}")
        # Check feature mismatch
        if processed_input.ndim == 1:
            processed_input = processed_input.reshape(1, -1)  # Ensure it has 2D shape
        
        if processed_input.shape[1] != expected_features:
            raise ValueError(f"Feature mismatch! Model expects {expected_features} features but received {processed_input.shape[1]}.")


        
        cluster = model.predict(processed_input)[0]
    
        # Cluster Profile
        cluster_profile = df.iloc[cluster].to_dict()
    
        # Display Results
        st.subheader("Health Assessment Results")
        st.write(f"**Assigned Health Cluster:** {cluster}")
        st.write(f"**Health Status:** {classify_health(cluster_profile)}")


if __name__ == '__main__':
    main()
