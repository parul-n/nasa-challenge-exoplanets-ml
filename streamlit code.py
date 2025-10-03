##This file is continued in association to the previous file- Model code.py
##This code is to convert the previous code into a interctive UI using Streamlit

##INSTALLING THE LIBRARY
!pip install streamlit pyngrok xgboost scikit-learn seaborn

#Setting up the software in Google Colab or any other notebook
#ps: this part might differ for different Python shells.
!streamlit run app.py --server.port 8501 & npx localtunnel --port 8501

#OPTIONAL- TO SUPPRESS WARNINGS IN COLAB FOR THE FORTHCOMING CODE
import os
os.environ["STREAMLIT_SUPPRESS_LOGS"] = "1"

##FINAL CODE FOR STREAMLIT (specially optimized for Hugging Face for faster space-building)
import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Exoplanet Identifier", layout="wide")
st.title("🌌 Exoplanet Identification Software")
st.markdown(
    "Enter Kepler mission parameters to predict if the candidate is an exoplanet, "
    "or upload a CSV for batch predictions."
)


# Lazy-load model function
model = None

def get_model():
    global model
    if model is None:
        import joblib
        model = joblib.load("final_model.pkl")
    return model



# Sidebar: Model Training
st.sidebar.header("XGBoost Hyperparameters & Training")
n_estimators = st.sidebar.slider("Number of Trees", 50, 1000, 100)
max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)

uploaded_train_file = st.sidebar.file_uploader("Upload CSV for training", type="csv")
if st.sidebar.button("Train New Model"):
    if uploaded_train_file:
        train_data = pd.read_csv(uploaded_train_file)
        X_train = train_data.iloc[:, :-1]
        y_train = train_data.iloc[:, -1]

        # Lazy import XGBoost
        from xgboost import XGBClassifier
        import joblib

        with st.spinner("Training model…"):
            clf = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            clf.fit(X_train, y_train)
            joblib.dump(clf, "final_model.pkl")
            st.success("✅ Model trained and saved successfully!")
            model = clf  # update global model

        # Feature importance (lazy import plotting libraries)
        import matplotlib.pyplot as plt
        import seaborn as sns

        st.subheader("Feature Importance")
        importance = clf.feature_importances_
        features = X_train.columns
        fig, ax = plt.subplots()
        sns.barplot(x=importance, y=features, ax=ax)
        st.pyplot(fig)
    else:
        st.sidebar.error("Upload a training CSV first!")



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
        clf = get_model()
        input_data = np.array([[koi_period, koi_duration, koi_depth, koi_ror,
                                koi_teq, koi_insol, koi_steff, koi_srad, koi_model_snr]])
        pred = clf.predict(input_data)[0]
        if pred == 1:
            st.success("✅ Predicted as EXOPLANET")
        else:
            st.error("❌ Predicted as NOT an exoplanet")
    except Exception as e:
        st.error(f"⚠️ Error: Train or load a model first! ({e})")



# Batch Prediction
st.subheader("Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload a CSV for batch prediction", type="csv", key="batch")
if uploaded_file:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(batch_data.head())

    if st.button("Predict Batch"):
        try:
            clf = get_model()
            batch_pred = clf.predict(batch_data)
            batch_data["Prediction"] = ["Exoplanet" if p==1 else "Not Exoplanet" for p in batch_pred]
            st.write(batch_data)
            st.success("✅ Batch prediction complete!")

            st.download_button(
                label="Download Predictions CSV",
                data=batch_data.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error: Train or load a model first! ({e})")


# Model Metrics (optional)
st.subheader("Model Metrics")
if st.checkbox("Show Model Metrics (requires X_test & y_test)"):
    try:
        X_test  # must be defined from your uploaded training split
        y_test
        clf = get_model()
        y_pred = clf.predict(X_test)

        from sklearn.metrics import classification_report, confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns

        st.text(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
    except Exception:
        st.error("X_test and y_test not defined. Upload training CSV and split dataset first.")

