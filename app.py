import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.mixture import GaussianMixture

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('SWAN data.csv')
    
    # Select columns (same as Colab)
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

    # Handle missing values (same as Colab)
    categorical_cols = ["RACE", "INSURAN10", "HLTHSER10", "SMOKERE10"]
    numerical_cols = ["DEPRESS10", "SLEEPQL10", "PRGNAN10", "DIABETE10", "HIGHBP10", "MDTALK10"]
    
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # Fill remaining missing
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Label encode categorical columns (expanded list from Colab)
    le = LabelEncoder()
    categorical_cols = [
        'PRGNAN10', 'LENGCYL10', 'ENDO10', 'ANEMIA10', 'DIABETE10', 'HIGHBP10', 
        'MIGRAIN10', 'BROKEBO10', 'OSTEOPO10', 'HEART110', 'CHOLST110', 'THYROI110',
        'INSULN110', 'ARTHRT110', 'FERTIL110', 'BCP110', 'REGVITA10', 'ONCEADA10',
        'ANTIOXI10', 'VITCOMB10', 'VITAMNA10', 'BETACAR10', 'VITAMNC10', 'VITAMND10',
        'VITAMNE10', 'CALCTUM10', 'IRON10', 'ZINC10', 'SELENIU10', 'FOLATE10', 
        'VTMSING10', 'EXERCIS10', 'YOGA10', 'DIETNUT10', 'SMOKERE10', 'HLTHSER10',
        'INSURAN10', 'DEPRESS10', 'SLEEPQL10', 'RACE'
    ]
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # Feature Engineering (NEW from Colab)
    df['CHRONIC_SCORE'] = df[['DIABETE10', 'HIGHBP10', 'HEART110', 'CHOLST110']].mean(axis=1)
    df['MENTAL_HEALTH_IDX'] = df[['DEPRESS10', 'SLEEPQL10', 'NERVES10']].mean(axis=1)
    df['HEALTH_SCORE'] = df[['CHRONIC_SCORE', 'MENTAL_HEALTH_IDX', 'OSTEOPO10', 'ANEMIA10', 'SLEEPQL10']].mean(axis=1)

    # Standard Scaling (REPLACED MinMax with Standard)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    df = pd.DataFrame(df_scaled, columns=df.columns)
    
    # PCA (NEW from Colab)
    pca = PCA(n_components=0.95)
    df_pca = pca.fit_transform(df_scaled)
    
    return df

# -------------------------------
# Load Pre-trained Model
# -------------------------------
@st.cache_data
def load_model():
    model = joblib.load('gmm_model.pkl')
    return model

# -------------------------------
# Improved Health Classification (Updated from Colab)
# -------------------------------
def classify_health(cluster_profile):
    """Improved classification with tiered thresholds"""
    risks = []

    # Tier 1: Immediate risks
    if cluster_profile['CHRONIC_SCORE'] > 0.8:
        risks.append("🚨 Critical Cardiac Risk")
    elif cluster_profile['CHRONIC_SCORE'] > 0.6:
        risks.append("⚠️ Moderate Cardiovascular Risk")

    # Mental Health
    if cluster_profile['MENTAL_HEALTH_IDX'] > 1.5:
        risks.append("🧠 Urgent Mental Health Needs")
    elif cluster_profile['MENTAL_HEALTH_IDX'] > 1.2:
        risks.append("😟 Mild Mental Health Support Needed")

    # Tier 2: Chronic conditions
    if cluster_profile['OSTEOPO10'] > 0.15:
        risks.append("🦴 High Osteoporosis Risk")
    elif cluster_profile['OSTEOPO10'] > 0.05:
        risks.append("🦴 Moderate Bone Health Risk")

    if cluster_profile['ANEMIA10'] > 1.3:
        risks.append("🩸 Severe Anemia Risk")
    elif cluster_profile['ANEMIA10'] > 1.1:
        risks.append("🩸 Mild Anemia Signs")

    # Tier 3: Lifestyle factors
    if cluster_profile['SLEEPQL10'] > 2.2:
        risks.append("💤 Critical Sleep Issues")
    elif cluster_profile['SLEEPQL10'] > 1.8:
        risks.append("💤 Mild Sleep Disturbances")

    # Final classification
    if cluster_profile['HEALTH_SCORE'] < 0.9 and len(risks) < 2:
        return "✅ Healthy Profile"
    elif not risks:
        return "🌟 Generally Healthy"
    else:
        return " | ".join(risks)

# -------------------------------
# Radar Chart Visualization (New from Colab)
# -------------------------------
def plot_enhanced_radar(cluster_profile):
    categories = ['CHRONIC_SCORE', 'MENTAL_HEALTH_IDX',
                 'OSTEOPO10', 'ANEMIA10', 'SLEEPQL10']
    
    normalized = (cluster_profile[categories] - cluster_profile[categories].min()) / \
                (cluster_profile[categories].max() - cluster_profile[categories].min())
    
    fig = px.line_polar(
        r=normalized.values,
        theta=categories,
        line_close=True,
        template='plotly_dark',
        title='Health Profile Radar Chart'
    )
    return fig

# -------------------------------
# Main App Function
# -------------------------------
def main():
    st.title("Women's Health Cluster Analysis Dashboard")
    
    # Load data and model
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
    
    # Create input dataframe with ENGINEERED FEATURES
    input_data = pd.DataFrame({
        'AGE10': [age],
        'DIABETE10': [diabetes],
        'HIGHBP10': [highbp],
        'DEPRESS10': [depression],
        'SLEEPQL10': [sleep_quality],
        'EXERCIS10': [exercise],
        # Add all other features with default values
        'PRGNAN10': [0], 'LENGCYL10': [0], 'ENDO10': [0], 'MIGRAIN10': [0],
        'BROKEBO10': [0], 'OSTEOPO10': [0], 'HEART110': [0], 'CHOLST110': [0],
        'THYROI110': [0], 'INSULN110': [0], 'NERVS110': [0], 'ARTHRT110': [0],
        'FERTIL110': [0], 'BCP110': [0], 'REGVITA10': [0], 'ONCEADA10': [0],
        'ANTIOXI10': [0], 'VITCOMB10': [0], 'VITAMNA10': [0], 'BETACAR10': [0],
        'VITAMNC10': [0], 'VITAMND10': [0], 'VITAMNE10': [0], 'CALCTUM10': [0],
        'IRON10': [0], 'ZINC10': [0], 'SELENIU10': [0], 'FOLATE10': [0],
        'VTMSING10': [0], 'YOGA10': [0], 'DIETNUT10': [0], 'SMOKERE10': [0],
        'MDTALK10': [0], 'HLTHSER10': [0], 'INSURAN10': [0], 'NERVES10': [0],
        'RACE': [0]
    })
    
    # Add engineered features to input data
    input_data['CHRONIC_SCORE'] = input_data[['DIABETE10', 'HIGHBP10', 'HEART110', 'CHOLST110']].mean(axis=1)
    input_data['MENTAL_HEALTH_IDX'] = input_data[['DEPRESS10', 'SLEEPQL10', 'NERVES10']].mean(axis=1)
    input_data['HEALTH_SCORE'] = input_data[['CHRONIC_SCORE', 'MENTAL_HEALTH_IDX', 'OSTEOPO10', 'ANEMIA10', 'SLEEPQL10']].mean(axis=1)

    
    # Prediction
    if st.sidebar.button("Predict"):
        # Preprocess input
        scaler = StandardScaler()
        processed_input = scaler.fit_transform(input_data)
        
        # Predict cluster
        cluster = model.predict(processed_input)[0]
        
        # Get cluster profile
        cluster_profile = df[df['Cluster'] == cluster].mean()
        
        # Display results
        st.subheader("Health Assessment Results")
        st.write(f"**Assigned Health Cluster:** {cluster}")
        
        health_status = classify_health(cluster_profile)
        st.write(f"**Health Status:** {health_status}")
        
        # Show radar chart
        st.plotly_chart(plot_enhanced_radar(cluster_profile))

        # Recommendations
        if "Cardiac" in health_status:
            st.warning("Recommended: Regular cardiac checkups and diet modification")
        if "Mental Health" in health_status:
            st.warning("Recommended: Counseling services and stress management")
        if "Osteoporosis" in health_status:
            st.warning("Recommended: Bone density screening and calcium supplements")

    # -------------------------------
    # Updated Visualizations
    # -------------------------------
    st.header("Population Health Insights")
    
    # Health Metric Distributions
    st.subheader("Health Metric Distributions")
    metric = st.selectbox("Select Metric", ['DIABETE10', 'HIGHBP10', 'DEPRESS10', 'SLEEPQL10', 'HEALTH_SCORE'])
    fig = px.histogram(df, x=metric, color='Cluster', nbins=20)
    st.plotly_chart(fig)
    
    # Cluster Health Profiles
    st.subheader("Cluster Health Profiles")
    cluster_metrics = df.groupby('Cluster')[['CHRONIC_SCORE', 'MENTAL_HEALTH_IDX', 'HEALTH_SCORE']].mean()
    st.bar_chart(cluster_metrics)
    
    # Feature Correlations
    st.subheader("Feature Correlations")
    corr = df.corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(corr, cmap='coolwarm')
    st.pyplot(plt)

if __name__ == '__main__':
    main()
