import warnings
warnings.filterwarnings("ignore")

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

# ------------------- Page Config -------------------
st.set_page_config(page_title="AQI Prediction App", layout="wide")

st.title("🌍 Air Quality Index (AQI) Prediction App")
st.markdown("This app uses **SVR** and **Random Forest** to predict AQI.")

# ------------------- Sidebar -------------------
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# ------------------- Main -------------------
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head())

        # ------------------- Preprocessing -------------------
        df = df.dropna()

        # Rename column if needed
        if 'AQI Value' in df.columns:
            df = df.rename(columns={'AQI Value': 'AQI'})

        # Check AQI column
        if 'AQI' not in df.columns:
            st.error("❌ Dataset must contain an 'AQI' column")
            st.stop()

        # Encode categorical column
        if 'Status' in df.columns:
            le = LabelEncoder()
            df['Status_Encoded'] = le.fit_transform(df['Status'])

        # ------------------- Visualization -------------------
        st.subheader("📈 Data Visualization")

        if st.checkbox("Show Correlation Heatmap"):
            numeric_df = df.select_dtypes(include=np.number)

            if numeric_df.shape[1] > 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
            else:
                st.warning("Not enough numeric columns for correlation heatmap")

        # ------------------- Features -------------------
        y = df['AQI']
        X = df.select_dtypes(include=np.number).drop('AQI', axis=1)

        if X.empty:
            st.error("❌ No numeric features found for training")
            st.stop()

        # ------------------- Scaling -------------------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ------------------- Train/Test Split -------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # ------------------- Models -------------------
        st.subheader("🤖 Model Training & Evaluation")

        # SVR
        svr = SVR()
        svr.fit(X_train, y_train)
        y_pred_svr = svr.predict(X_test)

        # Random Forest
        rfr = RandomForestRegressor(n_estimators=100, random_state=42)
        rfr.fit(X_train, y_train)
        y_pred_rfr = rfr.predict(X_test)

        # ------------------- Metrics -------------------
        col1, col2 = st.columns(2)

        with col1:
            st.info("SVR Performance")
            st.write(f"R² Score: {r2_score(y_test, y_pred_svr):.4f}")
            st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_svr)):.4f}")

        with col2:
            st.info("Random Forest Performance")
            st.write(f"R² Score: {r2_score(y_test, y_pred_rfr):.4f}")
            st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rfr)):.4f}")

        # ------------------- Prediction Section -------------------
        st.divider()
        st.subheader("🔮 Make a Prediction")

        user_inputs = []
        cols = st.columns(len(X.columns))

        for i, col_name in enumerate(X.columns):
            val = cols[i].number_input(
                f"{col_name}",
                value=float(X[col_name].mean())
            )
            user_inputs.append(val)

        if st.button("Predict AQI"):
            input_scaled = scaler.transform([user_inputs])
            prediction = rfr.predict(input_scaled)
            st.success(f"✅ Predicted AQI: {prediction[0]:.2f}")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

else:
    st.info("👈 Please upload a CSV file from the sidebar to start")
