## Streamlit app for Exoplanet Classification
## Associated to the trained model in file final_code.py
import streamlit as st
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Exoplanet Identifier", layout="wide")
st.title("🌌 Exoplanet Identification Software")
st.markdown(
    "Enter Kepler mission parameters to predict if the candidate is an exoplanet, "
    "or upload a CSV for batch predictions."
)



# Lazy-load model and scaler
model = None
scaler = None

def get_model():
    global model, scaler
    if model is None or scaler is None:
        import joblib
        model = joblib.load("final_model.pkl")
        scaler = joblib.load("scaler.pkl")
    return model, scaler




# Manual Input Prediction
st.subheader("Predict a Single Exoplanet Entry")

with st.form(key="single_predict"):
    koi_period = st.number_input("Orbital Period (days)", value=1.0)
    koi_duration = st.number_input("Transit Duration (hrs)", value=1.0)
    koi_depth = st.number_input("Transit Depth (ppm)", value=1.0)
    koi_ror = st.number_input("Planet-Star Radius Ratio", value=0.01)
    koi_teq = st.number_input("Equilibrium Temperature (K)", value=500.0)
    koi_insol = st.number_input("Insolation Flux", value=1.0)
    koi_steff = st.number_input("Stellar Effective Temperature (K)", value=5000.0)
    koi_srad = st.number_input("Stellar Radius (Solar Radii)", value=1.0)
    koi_model_snr = st.number_input("Model SNR", value=10.0)

    submit_single = st.form_submit_button("Predict Exoplanet")

if submit_single:
    try:
        clf, scaler = get_model()
        input_data = np.array([[koi_period, koi_duration, koi_depth, koi_ror,
                                koi_teq, koi_insol, koi_steff, koi_srad, koi_model_snr]])
        input_scaled = scaler.transform(input_data)
        pred = clf.predict(input_scaled)[0]

        if pred == 1:
            st.success("✅ Predicted as EXOPLANET")
        else:
            st.error("❌ Predicted as NOT an exoplanet")

    except Exception as e:
        st.error(f"Error: {e}")



# Batch Prediction
st.subheader("Batch Prediction from CSV")

uploaded_file = st.file_uploader("Upload a CSV for batch prediction", type="csv", key="batch")

if uploaded_file:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(batch_data.head())

    if st.button("Predict Batch"):
        try:
            clf, scaler = get_model()
            input_scaled = scaler.transform(batch_data)
            batch_pred = clf.predict(input_scaled)
            batch_data["Prediction"] = ["Exoplanet" if p == 1 else "Not Exoplanet" for p in batch_pred]

            st.write(batch_data)
            st.success("✅ Batch prediction complete!")

            st.download_button(
                label="Download Predictions CSV",
                data=batch_data.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error: {e}")



# Model Metrics 
st.subheader("Model Metrics")

if st.checkbox("Show Model Metrics (uses internal test sample)"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix

    TEST_CSV = "kepler_test_sample.csv"

    if os.path.exists(TEST_CSV):
        test_data = pd.read_csv(TEST_CSV)
        feature_cols = ["koi_period","koi_duration","koi_depth","koi_ror",
                        "koi_teq","koi_insol","koi_steff","koi_srad","koi_model_snr"]
        X_test = test_data[feature_cols]
        y_test = test_data["koi_pdisposition"]

        clf, scaler = get_model()
        X_test_scaled = scaler.transform(X_test)
        y_pred = clf.predict(X_test_scaled)

        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        st.text("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
    else:
        st.error("Internal test CSV not found!")
