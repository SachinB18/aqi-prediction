# aqi-prediction
Reproducible pipeline for AQI prediction using pollutant and weather data. 

## 📋 Project Overview

This project implements a machine learning-based Air Quality Index (AQI) prediction system that analyzes pollutant concentrations and weather data to predict air quality levels. The system provides both regression models (for predicting exact AQI values) and classification models (for categorizing air quality into different levels).

## 🎯 Key Features

- **Multiple ML Models**: Includes Linear Regression, Decision Tree, and Random Forest algorithms for both regression and classification tasks
- **Interactive Web Application**: Flask-based UI for making predictions with trained models
- **PPM to μg/m³ Conversion**: Automatic conversion of gas pollutant measurements using ideal gas law
- **Batch Predictions**: Support for CSV file uploads to process multiple predictions at once
- **Real-time Predictions**: Single prediction interface with immediate health advice
- **Model Performance Tracking**: Comprehensive metadata storage with R² scores, RMSE, and accuracy metrics
- **AQI Interpretation**: Automatic categorization into Good, Satisfactory, Moderate, Poor, and Very Poor with health advisories

## 🧪 Pollutants Monitored

The system tracks the following key air quality indicators: 
- **PM2.5** - Fine particulate matter (≤2.5 μm)
- **PM10** - Coarse particulate matter (≤10 μm)
- **NO₂** - Nitrogen Dioxide
- **CO** - Carbon Monoxide
- **SO₂** - Sulfur Dioxide
- **O₃** - Ozone

## 🏗️ Project Structure

```
aqi-prediction/
├── app.py                      # Flask web application
├── model_saver.py              # Model training and saving pipeline
├── AirQuality_Project_2.ipynb  # Jupyter notebook for analysis
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── saved_models/               # Directory for trained models (created at runtime)
    ├── *.pkl                   # Serialized model files
    └── model_metadata.json     # Model performance metrics
```

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/SachinB18/aqi-prediction.git
cd aqi-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training Models

Run the model training pipeline:
```bash
python model_saver.py
```

This will:
- Load and preprocess the air quality dataset
- Train multiple regression and classification models
- Save trained models to the `saved_models/` directory
- Generate performance metadata

### Running the Web Application

Start the Flask server:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## 📊 Model Performance

The system trains and evaluates multiple models: 

**Regression Models** (Predict exact AQI value):
- Linear Regression
- Decision Tree Regression
- Random Forest Regression

**Classification Models** (Predict AQI category):
- Decision Tree Classification
- Random Forest Classification

## 🎨 AQI Categories

| AQI Range | Category | Color | Health Impact |
|-----------|----------|-------|---------------|
| 0-50 | Good | Green | Minimal impact |
| 51-100 | Satisfactory | Yellow | Minor breathing discomfort to sensitive people |
| 101-200 | Moderate | Orange | Breathing discomfort to people with lung, heart disease |
| 201-300 | Poor | Red | Breathing discomfort to most people |
| 301-500 | Very Poor | Maroon | Respiratory illness to people on prolonged exposure |

## 🔬 Technical Details

- **Data Preprocessing**: MinMax scaling for features, validation of AQI values (0-500 range)
- **Unit Conversion**: Automatic PPM to μg/m³ conversion using ideal gas law at standard conditions (25°C, 1 atm)
- **Model Persistence**: Joblib serialization for efficient model storage and loading
- **Feature Selection**: Focus on the 6 most important pollutants that significantly impact air quality

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 

## 📝 License

This project is open source and available under the MIT License. 

## 👤 Author

**Sachin Bhabad**
- GitHub: [@SachinB18](https://github.com/SachinB18)

## 🙏 Acknowledgments

- Air quality data sourced from public environmental monitoring datasets
- Built with Flask, scikit-learn, and modern Python data science stack
