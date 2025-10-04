#ASSOCIATED AND USES DATA FROM KOI, AND CODE AND OTHER ATTRIBUTES FROM final_code.py
## STREAMLIT APP (Cloud-ready with auto-download)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import os
import gdown

st.set_page_config(page_title="NASA Exoplanet Classifier", layout="wide")
st.title("🌌 Exoplanet Classifier 🚀🪐")
st.markdown("""
Predict whether a candidate is a **Confirmed Planet** or **False Positive**.  
Enter the feature values below, choose a model manually or let the app auto-select the most confident prediction. Please review borderline entries.
""")

#Downloading a large file via Drive- can skip if not required
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
rf_model_path = os.path.join(MODEL_DIR, "rf_model.pkl")

#Drive direct download link
RF_DRIVE_URL = "https://drive.google.com/uc?export=download&id=1l3muIYhsH6LSuvNNrrQvVlGqZqHXxAI-"

if not os.path.exists(rf_model_path):
    with st.spinner("Downloading Random Forest model..."):
        gdown.download(RF_DRIVE_URL, rf_model_path, quiet=False)
        st.success("RF model downloaded!")


## Cached Model & Preprocessing Loading
@st.cache_resource
def load_models():
    rf_model = joblib.load(rf_model_path)
    # Load XGBoost model if needed; for small file you can include it in repo
    xgb_model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")
    return rf_model, xgb_model, scaler, le

rf_model, xgb_model, scaler, le = load_models()


#Load Test Data for ROC
@st.cache_data
def load_test_data():
    df = pd.read_csv("kepler_data.csv", comment="#")
    feature_cols = ["koi_period","koi_duration","koi_depth","koi_ror",
                    "koi_teq","koi_insol","koi_steff","koi_srad","koi_model_snr"]
    target_col = "koi_pdisposition"
    df = df[feature_cols + [target_col]]
    df[target_col] = le.transform(df[target_col])
    X_test = scaler.transform(df[feature_cols])
    y_test = df[target_col]
    return X_test, y_test, feature_cols, target_col

X_test, y_test, feature_cols, target_col = load_test_data()



## Inputs
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



## Prediction Logic
if submit_single:
    X_input = pd.DataFrame([[koi_period, koi_duration, koi_depth, koi_ror,
                             koi_teq, koi_insol, koi_steff, koi_srad, koi_model_snr]],
                           columns=feature_cols)
    X_scaled = scaler.transform(X_input)

    st.subheader("Prediction Mode")
    mode = st.radio("Select Mode", ["Choose Model", "Auto-select Best"])
    
    if mode == "Choose Model":
        model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost"])
        model = rf_model if model_choice == "Random Forest" else xgb_model
        pred_prob = model.predict_proba(X_scaled)[0,1]
        pred_class = model.predict(X_scaled)[0]
    else:
        rf_prob = rf_model.predict_proba(X_scaled)[0,1]
        xgb_prob = xgb_model.predict_proba(X_scaled)[0,1]
        if rf_prob >= xgb_prob:
            model = rf_model
            pred_prob = rf_prob
        else:
            model = xgb_model
            pred_prob = xgb_prob
        pred_class = model.predict(X_scaled)[0]



  
    ## Display Results
    st.subheader("Prediction Results")
    st.write(f"Predicted Class: **{le.inverse_transform([pred_class])[0]}**")
    st.write(f"Prediction Probability: **{pred_prob:.2f}**")
    st.info(f"Model used: **{'Random Forest' if model==rf_model else 'XGBoost'}**")

    if 0.4 < pred_prob < 0.6:
        st.warning("⚠️ Prediction is borderline. Consider reviewing feature values carefully.")

    # Probability bar chart
    classes = le.inverse_transform([0,1])
    probabilities = [1-pred_prob, pred_prob]
    fig, ax = plt.subplots()
    ax.bar(classes, probabilities, color=['gray','skyblue'])
    ax.set_ylim(0,1)
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Confidence")
    st.pyplot(fig)

    # ROC-AUC Curves
    if st.checkbox("Show ROC-AUC Curves for Test Data"):
        st.subheader("ROC-AUC Curves (Test Dataset)")
        rf_probs_test = rf_model.predict_proba(X_test)[:,1]
        xgb_probs_test = xgb_model.predict_proba(X_test)[:,1]
        fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs_test)
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs_test)
        
        fig2, ax2 = plt.subplots(figsize=(7,5))
        ax2.plot(fpr_rf, tpr_rf, label=f'Random Forest AUC={roc_auc_score(y_test, rf_probs_test):.3f}', color='blue')
        ax2.plot(fpr_xgb, tpr_xgb, label=f'XGBoost AUC={roc_auc_score(y_test, xgb_probs_test):.3f}', color='green')
        ax2.plot([0,1],[0,1],'--',color='gray')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC-AUC Curves on Test Data')
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

