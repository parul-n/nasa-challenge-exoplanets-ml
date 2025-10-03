
# 🌍 NASA Space Apps Challenge – ML Classifier 🚀
# nasa-challenge-exoplanets-ml
Developed for NASA Space Apps, this ML classifier preprocesses space-related data, tunes hyperparameters via Grid Search, and achieves ~84% accuracy. It includes label optimization, confusion matrix analysis, and classification reports for a reliable, interpretable, deployable solution for astronomers for identification of exoplanets using AI.

## Features
- Data preprocessing and cleaning
- Label encoding and optimization
- Hyperparameter tuning using Grid Search
- Model evaluation (accuracy, confusion matrix, classification reposrt)
- Easy deployment with Streamlit
- User-friendly interface for predictions

## **Dataset**
Kepler Object of Interest (KOI): https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative
**Features used for prediction:**
  - `koi_period` – Orbital Period (days)  
  - `koi_duration` – Transit Duration (hrs)  
  - `koi_depth` – Transit Depth (ppm)  
  - `koi_ror` – Planet-Star Radius Ratio  
  - `koi_teq` – Equilibrium Temperature (K)  
  - `koi_insol` – Insolation Flux  
  - `koi_steff` – Stellar Effective Temperature (K)  
  - `koi_srad` – Stellar Radius (Solar Radii)  
  - `koi_model_snr` – Model SNR  


## **Technologies Used**
Python 3.x,
scikit-learn,
XGBoost,
Logistic Regression,
Grid Search,
pandas, numpy,
joblib,
Streamlit,

## **Algorithm:** XGBoost Classifier  
- **Hyperparameters:**  
  - n_estimators: 300  
  - max_depth: 6  
  - learning_rate: 0.05  
  - subsample: 0.8  
  - colsample_bytree: 0.8  
- Trained with preprocessed Kepler dataset and saved as `final_model.pkl`.


## **By:**
Parul Nagarwal
(Team: nasa spons0rers- NASA SPACE APPS CHALLENGE 2025)



