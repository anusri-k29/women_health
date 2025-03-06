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
    
    # Use original 44 features that match the trained model
    selected_columns = [
        'AGE10', 'PRGNAN10', 'LENGCYL10', 'ENDO10', 'ANEMIA10', 'DIABETE10', 
        'HIGHBP10', 'MIGRAIN10', 'BROKEBO10', 'OSTEOPO10', 'HEART110', 'CHOLST110',
        'THYROI110', 'INSULN110', 'NERVS110', 'ARTHRT110', 'FERTIL110', 'BCP110',
        'REGVITA10', 'ONCEADA10', 'ANTIOXI10', 'VITCOMB10', 'VITAMNA10', 'BETACAR10',
        'VITAMNC10', 'VITAMND10', 'VITAMNE10', 'CALCTUM10', 'IRON10', 'ZINC10',
        'SELENIU10', 'FOLATE10', 'VTMSING10', 'EXERCIS10', 'YOGA10', 'DIETNUT10',
        'SMOKERE10', 'MDTALK10', 'HLTHSER10', 'INSURAN10', 'NERVES10', 'DEPRESS10',
        'SLEEPQL10', 'RACE'  # Added back missing columns
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
    
    return pd.DataFrame(df_scaled, columns=df.columns), scaler, selected_columns

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

    # Debugging: Display number of clusters in the model
    st.write(f"**Number of clusters in the model:** {model.n_components}")

    # Sidebar Inputs
    st.sidebar.header("Patient Health Assessment")
    user_inputs = {col: 0.0 for col in df.columns}  # Initialize all features
    
    # Update with actual inputs
    user_inputs.update({
        'AGE10': st.sidebar.slider("Age", 20, 80, 40),
        'DIABETE10': st.sidebar.selectbox("Diabetes", [0, 1]),
        'HIGHBP10': st.sidebar.selectbox("High Blood Pressure", [0, 1]),
        'DEPRESS10': st.sidebar.slider("Depression Score", 0.0, 5.0, 2.5),
        'SLEEPQL10': st.sidebar.slider("Sleep Quality (1=good, 5=poor)", 1, 5, 3),
        'EXERCIS10': st.sidebar.slider("Exercise Frequency (days/week)", 0, 7, 3)
    })
    
    if st.sidebar.button("Predict"):
        # Create input DataFrame with ALL features
        input_data = pd.DataFrame([user_inputs])[df.columns]  # Ensure correct order
        
        # Debugging: Check input data shape
        st.write(f"**Shape of input data before scaling:** {input_data.shape}")

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Debugging: Check if input shape matches model expectations
        if input_scaled.shape[1] != model.n_features_in_:
            st.error(f"Feature mismatch! Model expects {model.n_features_in_} features but received {input_scaled.shape[1]}.")
            return

        # Predict cluster
        assigned_cluster = model.predict(input_scaled)[0]
        
        # Debugging: Show probability distribution over clusters
        cluster_probs = model.predict_proba(input_scaled)
        st.write(f"**Cluster Probabilities:** {cluster_probs}")

        # Cluster interpretation messages
        cluster_messages = {
            0: "Cluster 0: Active lifestyle, lower health risks.",
            1: "Cluster 1: Moderate risk, needs better sleep & diet.",
            2: "Cluster 2: Higher risk group, needs medical attention.",
            3: "Cluster 3: Needs stress management & regular checkups."
        }

        st.subheader("Health Assessment Results")
        st.write(f"**Assigned Health Cluster:** {assigned_cluster}")
        st.write(cluster_messages.get(assigned_cluster, "Unknown cluster."))
# PCA for visualization
def plot_clusters(df, model):
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df)
    cluster_labels = model.predict(df)

    fig = px.scatter(x=df_pca[:, 0], y=df_pca[:, 1], color=cluster_labels.astype(str),
                     labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'},
                     title="GMM Cluster Visualization (PCA)")
    st.plotly_chart(fig)

plot_clusters(df, model)
        
if __name__ == '__main__':
    main()
