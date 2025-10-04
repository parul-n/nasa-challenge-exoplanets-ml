##STREAMLIT APP
#ASSOCIATED AND USES DATA FROM KOI, AND CODE AND OTHER ATTRIBUTES FROM final_code.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


##Load trained models & preprocessing
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")



##load test data for ROC curves
test_data = pd.read_csv("kepler_data.csv", comment="#")
feature_cols = ["koi_period","koi_duration","koi_depth","koi_ror",
                "koi_teq","koi_insol","koi_steff","koi_srad","koi_model_snr"]
target_col = "koi_pdisposition"
test_data = test_data[feature_cols + [target_col]]
test_data[target_col] = le.transform(test_data[target_col])
X_test = scaler.transform(test_data[feature_cols])
y_test = test_data[target_col]

st.set_page_config(page_title="NASA Exoplanet Classifier", layout="wide")
st.title("🌌 NASA Exoplanet Classifier 🚀")
st.markdown("""
Predict whether a candidate is a **Confirmed Planet** or **False Positive**.  
Enter the feature values below, choose a model manually or let the app auto-select the most confident prediction. Please review borderline entries.
""")



##Form-based input
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



##Prediction logic
if submit_single:
    X_input = pd.DataFrame([[
        koi_period, koi_duration, koi_depth, koi_ror,
        koi_teq, koi_insol, koi_steff, koi_srad, koi_model_snr
    ]], columns=feature_cols)
    
    X_scaled = scaler.transform(X_input)

    st.subheader("Prediction Mode")
    mode = st.radio("Select Mode", ["Choose Model", "Auto-select Best"])
    if mode == "Choose Model":
        model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost"])
        model = rf_model if model_choice == "Random Forest" else xgb_model
        pred_prob = model.predict_proba(X_scaled)[0, 1]
        pred_class = model.predict(X_scaled)[0]
    else:
        rf_prob = rf_model.predict_proba(X_scaled)[0, 1]
        xgb_prob = xgb_model.predict_proba(X_scaled)[0, 1]
        if rf_prob >= xgb_prob:
            model = rf_model
            pred_prob = rf_prob
        else:
            model = xgb_model
            pred_prob = xgb_prob
        pred_class = model.predict(X_scaled)[0]

   

    ##Display Results
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



    ##ROC-AUC curves
    if st.checkbox("Show ROC-AUC Curves for Test Data"):
        st.subheader("ROC-AUC Curves (Test Dataset)")
        rf_probs_test = rf_model.predict_proba(X_test)[:, 1]
        xgb_probs_test = xgb_model.predict_proba(X_test)[:, 1]
        fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs_test)
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs_test)
        plt.figure(figsize=(7,5))
        plt.plot(fpr_rf, tpr_rf, label=f'Random Forest AUC={roc_auc_score(y_test, rf_probs_test):.3f}')
        plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost AUC={roc_auc_score(y_test, xgb_probs_test):.3f}')
        plt.plot([0,1],[0,1],'--',color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-AUC Curves on Test Data')
        plt.legend()
        plt.grid()
        st.pyplot(plt)
