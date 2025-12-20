"""
AQI Prediction Web Application
Flask-based UI for loading saved models and making AQI predictions
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
import traceback

app = Flask(__name__)
app.secret_key = 'aqi_prediction_secret_key'

def ppm_to_ugm3(ppm_value, molecular_weight, temperature=25, pressure=1013.25):
    """
    Convert ppm to μg/m³ using ideal gas law
    Default conditions: 25°C, 1 atm pressure
    """
    # Convert temperature to Kelvin
    temp_k = temperature + 273.15
    
    # Ideal gas constant (L·atm/mol·K)
    R = 0.0821
    
    # Convert ppm to μg/m³
    # μg/m³ = (ppm × MW × P × 1000) / (R × T)
    ugm3 = (ppm_value * molecular_weight * pressure * 1000) / (R * temp_k)
    
    return ugm3

def get_molecular_weights():
    """Return molecular weights for common pollutants (for PPM conversion)"""
    return {
        'NO2': 46.01,   # g/mol
        'CO': 28.01,    # g/mol  
        'SO2': 64.07,   # g/mol
        'O3': 48.00     # g/mol
    }

class AQIPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.metadata = {}
        self.feature_columns = []
        self.load_models()
    
    def load_models(self):
        """Load all saved models and metadata"""
        try:
            # Load metadata
            if os.path.exists('saved_models/model_metadata.json'):
                with open('saved_models/model_metadata.json', 'r') as f:
                    self.metadata = json.load(f)
                    self.feature_columns = self.metadata.get('feature_columns', [])
            
            # Load models
            model_files = [f for f in os.listdir('saved_models') if f.endswith('.pkl')]
            
            for model_file in model_files:
                model_name = model_file.replace('.pkl', '')
                model_path = os.path.join('saved_models', model_file)
                
                try:
                    model = joblib.load(model_path)
                    
                    if 'scaler' in model_name or 'encoder' in model_name:
                        self.scalers[model_name] = model
                    else:
                        self.models[model_name] = model
                    
                    print(f"Loaded: {model_name}")
                except Exception as e:
                    print(f"Error loading {model_name}: {e}")
            
            print(f"Loaded {len(self.models)} models and {len(self.scalers)} scalers")
            
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def get_available_models(self):
        """Return list of available models with metadata"""
        available = []
        for model_name in self.models.keys():
            model_info = self.metadata.get('models', {}).get(model_name, {})
            available.append({
                'name': model_name,
                'type': model_info.get('type', 'unknown'),
                'target': model_info.get('target', 'unknown'),
                'performance': self.get_model_performance(model_name, model_info)
            })
        return available
    
    def get_model_performance(self, model_name, model_info):
        """Get performance metrics for a model"""
        if model_info.get('type') == 'regression':
            return f"R² = {model_info.get('r2_score', 0):.3f}, RMSE = {model_info.get('rmse', 0):.3f}"
        elif model_info.get('type') == 'classification':
            return f"Accuracy = {model_info.get('accuracy', 0):.3f}"
        return "N/A"

    def predict(self, model_name, input_data):
        """Make prediction using specified model"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model = self.models[model_name]
            
            # Convert ppb to μg/m³ for gas pollutants
            converted_data = input_data.copy()
            molecular_weights = get_molecular_weights()
            
            # Convert ppm values to μg/m³
            ppm_pollutants = ['NO2', 'CO', 'SO2', 'O3']
            for pollutant in ppm_pollutants:
                if f"{pollutant}_ppm" in converted_data:
                    ppm_value = converted_data[f"{pollutant}_ppm"]
                    if pollutant in molecular_weights and ppm_value > 0:
                        ugm3_value = ppm_to_ugm3(ppm_value, molecular_weights[pollutant])
                        converted_data[pollutant] = ugm3_value
                    # Remove the ppm key
                    del converted_data[f"{pollutant}_ppm"]
                elif pollutant in converted_data:
                    # If direct μg/m³ value is provided, use it as is
                    pass
            
            # Prepare input data
            input_df = pd.DataFrame([converted_data])
            
            # Ensure all feature columns are present
            for col in self.feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0  # Default value for missing features
            
            # Reorder columns to match training data
            input_df = input_df[self.feature_columns]
            
            # Scale input data if scaler is available
            if 'feature_scaler' in self.scalers:
                scaler = self.scalers['feature_scaler']
                input_scaled = scaler.transform(input_df)
                input_df = pd.DataFrame(input_scaled, columns=self.feature_columns)
            
            # Make prediction
            raw_prediction = model.predict(input_df)[0]
            
            # Validate and constrain prediction to realistic range
            if isinstance(raw_prediction, (int, float, np.number)):
                # For regression models, constrain AQI to 0-500 range
                prediction = max(0, min(500, float(raw_prediction)))
                if abs(raw_prediction - prediction) > 1:
                    print(f"WARNING: Constrained prediction from {raw_prediction:.2f} to {prediction:.2f}")
            else:
                prediction = raw_prediction
            
            # Get model type and additional info
            model_info = self.metadata.get('models', {}).get(model_name, {})
            model_type = model_info.get('type', 'unknown')
            
            result = {
                'prediction': float(prediction) if isinstance(prediction, (int, float, np.number)) else str(prediction),
                'model_name': model_name,
                'model_type': model_type
            }
            
            # Add interpretation for classification models
            if model_type == 'classification' and 'label_encoder' in self.scalers:
                # If prediction is encoded, decode it
                try:
                    if isinstance(prediction, (int, np.integer)):
                        le = self.scalers['label_encoder']
                        decoded_prediction = le.inverse_transform([prediction])[0]
                        result['prediction'] = decoded_prediction
                except:
                    pass
            
            # Add AQI interpretation for regression models
            if model_type == 'regression':
                aqi_value = result['prediction']
                result['interpretation'] = self.interpret_aqi(aqi_value)
            
            return result
            
        except Exception as e:
            return {'error': str(e), 'traceback': traceback.format_exc()}

    def interpret_aqi(self, aqi_value):
        """Interpret AQI value into category and health advice"""
        if aqi_value <= 50:
            return {
                'category': 'Good',
                'color': 'green',
                'health_advice': 'Air quality is satisfactory and poses little or no health risk.'
            }
        elif aqi_value <= 100:
            return {
                'category': 'Satisfactory',
                'color': 'yellow',
                'health_advice': 'Air quality is acceptable for most people. Unusually sensitive individuals may experience minor symptoms.'
            }
        elif aqi_value <= 200:
            return {
                'category': 'Moderate',
                'color': 'orange',
                'health_advice': 'Members of sensitive groups may experience health effects. The general public is not likely to be affected.'
            }
        elif aqi_value <= 300:
            return {
                'category': 'Poor',
                'color': 'red',
                'health_advice': 'Everyone may begin to experience health effects. Members of sensitive groups may experience more serious effects.'
            }
        else:
            return {
                'category': 'Very Poor',
                'color': 'maroon',
                'health_advice': 'Health warnings of emergency conditions. The entire population is more likely to be affected.'
            }

# Initialize predictor
predictor = AQIPredictor()

@app.route('/')
def index():
    """Home page"""
    available_models = predictor.get_available_models()
    print(f"DEBUG: Available models: {len(available_models)}")
    print(f"DEBUG: Feature columns: {predictor.feature_columns}")
    return render_template('index.html', 
                         models=available_models, 
                         features=predictor.feature_columns)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.get_json()
        
        model_name = data.get('model_name')
        input_data = data.get('input_data', {})
        
        if not model_name:
            return jsonify({'error': 'Model name is required'}), 400
        
        # Make prediction
        result = predictor.predict(model_name, input_data)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/models')
def models():
    """Display model information"""
    available_models = predictor.get_available_models()
    return render_template('models.html', models=available_models)

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Handle batch predictions from CSV upload"""
    try:
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(url_for('index'))
        
        file = request.files['file']
        model_name = request.form.get('model_name')
        
        if file.filename == '':
            flash('No file selected')
            return redirect(url_for('index'))
        
        if not model_name:
            flash('No model selected')
            return redirect(url_for('index'))
        
        # Read CSV file
        df = pd.read_csv(file)
        
        # Make predictions for each row
        predictions = []
        for idx, row in df.iterrows():
            input_data = row.to_dict()
            result = predictor.predict(model_name, input_data)
            
            if 'error' not in result:
                predictions.append({
                    'row': idx + 1,
                    'prediction': result['prediction'],
                    'model_type': result['model_type']
                })
            else:
                predictions.append({
                    'row': idx + 1,
                    'prediction': 'Error',
                    'error': result['error']
                })
        
        return render_template('batch_results.html', 
                             predictions=predictions, 
                             model_name=model_name)
        
    except Exception as e:
        flash(f'Error processing file: {str(e)}')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

