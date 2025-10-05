
# ML Exoplanet Classifier
# Nasa Space Apps Challenge 2025
Developed for NASA Space Apps, this ML classifier preprocesses space-related data, trained via XGBoost. tunes hyperparameters via Grid Search, and achieves ~83% accuracy and 92% discriination factor (ROC-AUC). It includes label optimization, confusion matrix analysis, and classification reports, Feature Importance, ROC-AUC Curve for a reliable, interpretable, deployable solution for astronomers for identification of exoplanets using AI.

I have tried to experiment with different Machine Learning algorithms like, Random Forest, and XGBoost. Each gave varying results but the disparities in the evaluation were not huge. Ultimately, I preferred XGBoost which was best suited for our final model, and deployed it in our final software.

The ML techniques, algorithms, and approaches for this project align with the following papers:
1. MNRAS: Exoplanet detection using machine learning- https://academic.oup.com/mnras/article/513/4/5505/6472249
2. MDPI: Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification- https://www.mdpi.com/2079-9292/13/19/3950

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



