#ASSOCIATED AND USES DATA FROM KOI, AND CODE AND OTHER ATTRIBUTES FROM final_code.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score


##PAGE SETUP
st.set_page_config(page_title="Exoplanet Classifier 🚀🪐", layout="wide")

st.title("Exoplanet ML Classifier 🚀")
st.markdown("This ML app predicts whether a celestial body is an **Exoplanet** or **Non-Exoplanet** based on NASA’s dataset.")
st.markdown("Please be mindful of reviewing baseline entries.")


##LOAD MODEL FROM GOOGLE DRIVE (skip this if not uploading large files via Drive)
with st.spinner("Fetching model and dependencies from Google Drive and downloading files..."):
    file_id = "1l3muIYhsH6LSuvNNrrQvVlGqZqHXxAI-" 
    model_path = "rf_model.pkl"

    if not os.path.exists(model_path):
        gdown.download(f"https://drive.google.com/uc?id=1l3muIYhsH6LSuvNNrrQvVlGqZqHXxAI-", model_path, quiet=False)

scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
model = joblib.load(model_path)
st.success("Model loaded successfully!!")



## INPUT SECTION
st.header("INPUT FEATURES")

st.write("Enter or upload the input data for prediction:")

option = st.radio("Choose input method:", ["Manual Input", "Upload CSV", "Try Sample Test Cases"])

if option == "Manual Input":
    st.subheader("Predict a Single Exoplanet Entry")

    koi_period = st.number_input("Orbital Period (days)", value=1.0, min_value=0.0)
    koi_duration = st.number_input("Transit Duration (hrs)", value=1.0, min_value=0.0)
    koi_depth = st.number_input("Transit Depth (ppm)", value=1.0, min_value=0.0)
    koi_ror = st.number_input("Planet-Star Radius Ratio", value=0.01, min_value=0.0)
    koi_teq = st.number_input("Equilibrium Temperature (K)", value=500.0, min_value=0.0)
    koi_insol = st.number_input("Insolation Flux", value=1.0, min_value=0.0)
    koi_steff = st.number_input("Stellar Effective Temperature (K)", value=5000.0, min_value=0.0)
    koi_srad = st.number_input("Stellar Radius (Solar Radii)", value=1.0, min_value=0.0)
    koi_model_snr = st.number_input("Model SNR", value=10.0, min_value=0.0)

    if st.button("🔮 Predict Exoplanet"):
        input_data = np.array([[koi_period, koi_duration, koi_depth, koi_ror,
                                koi_teq, koi_insol, koi_steff, koi_srad, koi_model_snr]])
        
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        predicted_class = label_encoder.inverse_transform([prediction])[0]

        st.success(f"🪐 Predicted Class: **{predicted_class}**")

elif option == "Upload CSV":
    st.subheader("Upload Your CSV File")
    uploaded_file = st.file_uploader("Upload CSV with same features as training data", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.dataframe(data.head())

        # Check if columns match
        required_features = ["koi_period","koi_duration","koi_depth","koi_ror",
                     "koi_teq","koi_insol","koi_steff","koi_srad","koi_model_snr"]

        if set(required_features).issubset(data.columns):
            input_data = data[required_features].values

            # Predict Button for CSV Upload
            if st.button("🔮 Predict Uploaded Data"):
                scaled_input = scaler.transform(input_data)
                predictions = model.predict(scaled_input)
                predicted_labels = label_encoder.inverse_transform(predictions)
                data["Predicted Class"] = predicted_labels

                st.subheader("✅ Predictions")
                st.dataframe(data[["Predicted Class"] + required_features])
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Predictions as CSV", csv, "exoplanet_predictions.csv", "text/csv")
        else:
            st.error("❌ Uploaded file does not contain the required feature columns.")
            

else:  # Test Cases Mode
    st.subheader("Sample Test Cases (Simulated KOI Entries)")

    sample_data = pd.DataFrame({
        "koi_period": [1.5, 10.2, 250.6, 50.1, 365.3],
        "koi_duration": [2.5, 1.2, 3.6, 2.1, 10.5],
        "koi_depth": [100, 200, 50, 500, 150],
        "koi_ror": [0.02, 0.04, 0.01, 0.03, 0.02],
        "koi_teq": [800, 1200, 500, 900, 300],
        "koi_insol": [1.5, 2.1, 0.3, 1.0, 0.9],
        "koi_steff": [5500, 6000, 4800, 7200, 5600],
        "koi_srad": [0.9, 1.2, 0.8, 1.5, 1.0],
        "koi_model_snr": [20, 15, 8, 25, 12]
    })

    st.markdown("Below are **five synthetic KOI entries** representing different exoplanet candidates. You can run batch predictions on them:")

    st.dataframe(sample_data.style.format(precision=2))

    if st.button("🔮 Run Predictions on Sample Data"):
        X_sample_scaled = scaler.transform(sample_data)
        preds = model.predict(X_sample_scaled)
        pred_labels = label_encoder.inverse_transform(preds)

        sample_data["Predicted Class"] = pred_labels
        sample_data["Confidence"] = model.predict_proba(X_sample_scaled)[:, 1].round(2)

        st.subheader("🪐 Prediction Results")
        st.dataframe(sample_data.style.background_gradient(subset=["Confidence"], cmap="Blues").format(precision=2))



## MODEL METRICS & SHAP INSIGHTS
st.header("📈 Model Metrics & Model Insights")

if st.checkbox("Show Model Metrics (using real test data)"):

    # 2. Load model, scaler, label encoder, test data
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("rf_model.pkl")  # or xgb_model.pkl
    le = joblib.load("label_encoder.pkl")
    X_test, y_test = joblib.load("test_data.pkl")
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    # 2. Define readable feature names
    feature_name_map = {
        "koi_period": "Orbital Period (days)",
        "koi_duration": "Transit Duration (hrs)",
        "koi_depth": "Transit Depth (ppm)",
        "koi_ror": "Planet-Star Radius Ratio",
        "koi_teq": "Equilibrium Temperature (K)",
        "koi_insol": "Insolation Flux",
        "koi_steff": "Stellar Effective Temperature (K)",
        "koi_srad": "Stellar Radius (Solar Radii)",
        "koi_model_snr": "Model SNR"
    }
    features = list(feature_name_map.keys())
    readable_features = [feature_name_map[f] for f in features]

    # --- Class Distribution ---
    with st.expander("Class Distribution"):
        st.write(pd.Series(y_test).value_counts())

    # --- Classification Report ---
    with st.expander("Classification Report"):
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose())

    # --- Confusion Matrix ---
    with st.expander("Confusion Matrix"):
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig, clear_figure=True)

    # --- ROC-AUC Curve ---
    with st.expander("ROC-AUC Curve"):
        if hasattr(model, "predict_proba") and len(np.unique(y_test)) > 1:
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = roc_auc_score(y_test, y_proba)

            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
            ax2.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.set_title("Receiver Operating Characteristic (ROC) Curve")
            ax2.legend(loc="lower right")
            st.pyplot(fig2, clear_figure=True)
        else:
            st.warning("ROC-AUC plot unavailable: either the model does not support probability predictions or the test set has only one class.")

    with st.expander("Feature Importance & SHAP Explanations"):

        st.subheader("Feature Importance")
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': readable_features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        st.dataframe(importance_df)

        fig3, ax3 = plt.subplots(figsize=(7,5))
        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax3, palette="viridis")
        ax3.set_title("Feature Importance")
        st.pyplot(fig3, clear_figure=True)

        
# #FOOTER
st.markdown("---")
st.markdown("Developed for **NASA Space Apps Challenge 2025** 🌌 | Team: nasa spons0rers")




















