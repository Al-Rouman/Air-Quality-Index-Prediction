import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Page configuration
st.set_page_config(page_title="AQI Prediction App", layout="wide")

st.title("🌍 Air Quality Index (AQI) Prediction App")
st.markdown("Ei app-ti SVR ebong Random Forest bebohar kore AQI predict kore.")

# Sidebar - File Upload
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_select("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Data Preprocessing
    df = df.dropna()
    if 'AQI Value' in df.columns:
        df = df.rename(columns={'AQI Value': 'AQI'})
    
    # Label Encoding for categorical data
    if 'Status' in df.columns:
        le = LabelEncoder()
        df['Status_Encoded'] = le.fit_transform(df['Status'])

    # Correlation Plot
    if st.checkbox("Show Correlation Heatmap"):
        st.subheader("Correlation Analysis")
        fig, ax = plt.subplots(figsize=(10, 8))
        numeric_df = df.select_dtypes(include=np.number)
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # Features and Target
    y = df['AQI']
    X = df.select_dtypes(include=np.number).drop('AQI', axis=1, errors='ignore')

    if not X.empty:
        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Model Training
        st.subheader("Model Training & Evaluation")
        
        # SVR
        svr = SVR()
        svr.fit(X_train, y_train)
        y_pred_svr = svr.predict(X_test)
        
        # Random Forest
        rfr = RandomForestRegressor(n_estimators=100)
        rfr.fit(X_train, y_train)
        y_pred_rfr = rfr.predict(X_test)

        # Metrics display
        col1, col2 = st.columns(2)
        with col1:
            st.info("SVR Performance")
            st.write(f"R² Score: {r2_score(y_test, y_pred_svr):.4f}")
            st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_svr)):.4f}")
            
        with col2:
            st.info("Random Forest Performance")
            st.write(f"R² Score: {r2_score(y_test, y_pred_rfr):.4f}")
            st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rfr)):.4f}")

        # Prediction UI
        st.divider()
        st.subheader("🔮 Make a Prediction")
        
        # User input for prediction based on features
        user_inputs = []
        cols = st.columns(len(X.columns))
        for i, col_name in enumerate(X.columns):
            val = cols[i].number_input(f"Enter {col_name}", value=float(X[col_name].mean()))
            user_inputs.append(val)
        
        if st.button("Predict AQI"):
            input_scaled = scaler.transform([user_inputs])
            prediction = rfr.predict(input_scaled)
            st.success(f"Predicted AQI: {prediction[0]:.2f}")
    else:
        st.error("Dataset e porjapto numeric features nei.")
else:
    st.info("Please upload a CSV file from the sidebar to begin.")
