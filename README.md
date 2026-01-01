# AQI Prediction — ML + Flask Web UI

A complete system to predict Air Quality Index (AQI) from pollutant concentrations, with saved machine learning models and a lightweight Flask web interface.

## Overview
- Trains and saves multiple models for AQI prediction.
- Loads saved models for real-time inference via a web UI.
- Single prediction and batch CSV prediction.
- Clear unit guidance and automatic gas unit conversion.
- Clean, responsive Bootstrap UI.

## Tech Stack
- Backend: Flask, scikit-learn, pandas, numpy
- Frontend: HTML5, CSS3, Bootstrap, JavaScript
- Analysis: Jupyter Notebook (`AirQuality_Project_2.ipynb`)

## Project Structure

AQI DA/
├── app.py
├── model_saver.py
├── requirements.txt
├── AirQuality_Project_2.ipynb
├── saved_models/
│ └── model_metadata.json
├── templates/
│ ├── base.html
│ ├── index.html
│ ├── models.html
│ └── batch_results.html
└── aqi-prediction-project/ (modular code, tests, configs)


## Setup (Windows)
1. Install Python 3.9+ and Git (optional).
2. Create and activate a virtual environment, then install dependencies:
```powershell
cd "c:\Users\LOQ 15IRX9\Downloads\AQI DA"
python -m venv .venv
[Activate.ps1](http://_vscodecontentref_/4)
pip install -r [requirements.txt](http://_vscodecontentref_/5)

```

Train and Save Models
Run the trainer to generate models and metadata under saved_models/:
python [model_saver.py](http://_vscodecontentref_/6)


Outputs:

Model files (*.pkl/*.joblib)
saved_models/model_metadata.json (metrics, features)
Run the Web App
Start the Flask server and open the UI:

python [app.py](http://_vscodecontentref_/7)

URL: http://127.0.0.1:5000
Using the App
Single Prediction:

Choose a model (e.g., Random Forest Regression).
Enter pollutant values (see Units below).
Submit to get numeric AQI and a category interpretation.
Batch Prediction:

Upload a CSV with required pollutant columns.
Select a model and run batch inference.
Download results as CSV.

PM2.5,PM10,NO2,CO,SO2,O3
35.2,68.5,0.025,1.8,0.012,0.060
15.6,32.1,0.019,0.4,0.008,0.035

Features and Target
Features (6 core pollutants):
Particulates: PM2.5, PM10
Gases: NO2, CO, SO2, O3
Targets:
Regression: numeric AQI
Classification: AQI_Category (Good, Moderate, Poor)
PM2.5 generally has the strongest influence on AQI, followed by PM10.

Units and Conversion
Particulates:
PM2.5, PM10: input in μg/m³ (micrograms per cubic meter)
Gases:
NO2, CO, SO2, O3: input in PPM (parts per million)
The app converts PPM to μg/m³ internally using the ideal gas law at standard conditions (25°C, 1 atm).
Tips:

Use accurate measurements from reliable sources (government stations or calibrated sensors).
Ensure units match the UI fields for best accuracy.
Recommended Models
Regression (numeric AQI): Random Forest Regression (high accuracy and robustness)
Classification (AQI category): Random Forest Classification
Troubleshooting
“Models not found”: Run python model_saver.py to create saved_models/ contents.
“Unrealistic AQI”: Verify units and input values; gas inputs should be in PPM, particulates in μg/m³.
CSV errors: Ensure header names match the example and values are numeric.
