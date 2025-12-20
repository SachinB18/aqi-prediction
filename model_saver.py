"""
Model Training and Saving Script for AQI Prediction
This script trains multiple models and saves them for use in the UI
"""

import pandas as pd
import numpy as np
import pickle
import joblib
import json
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error, classification_report
import warnings
warnings.filterwarnings("ignore")

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

class AQIModelSaver:
    def __init__(self, data_path="C:\\Users\\LOQ 15IRX9\\Downloads\\city_day.csv"):
        self.data_path = data_path
        self.models = {}
        self.scalers = {}
        self.metadata = {}
        
        # Create directories
        os.makedirs('saved_models', exist_ok=True)
        
    def load_and_preprocess_data(self):
        """Load and preprocess the data"""
        print("Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv(self.data_path)
        df = df.dropna()
        
        # Clean column names
        df.columns = [col.strip() for col in df.columns]
        
        # Convert Date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Select only the most important features for AQI prediction
        # These are the main pollutants that significantly affect air quality
        important_features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
        
        # Filter to only include features that exist in the dataset
        available_features = [col for col in important_features if col in df.columns]
        
        # If none of the important features are available, fall back to any numeric columns
        if not available_features:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            available_features = [col for col in numeric_cols if col not in ['AQI']][:6]  # Limit to 6 features
        
        # Initialize and fit scaler for features only (NOT for AQI target)
        scaler = MinMaxScaler()
        if len(available_features) > 0:
            df[available_features] = scaler.fit_transform(df[available_features])
            self.scalers['feature_scaler'] = scaler
            
        # Do NOT scale the AQI target - keep it in original units
        
        # Store feature names for later use
        self.feature_columns = available_features
        
        # Validate AQI values and remove unrealistic ones
        if 'AQI' in df.columns:
            print(f"AQI statistics before cleaning:")
            print(f"  Min AQI: {df['AQI'].min()}")
            print(f"  Max AQI: {df['AQI'].max()}")
            print(f"  Mean AQI: {df['AQI'].mean():.2f}")
            
            # Remove unrealistic AQI values (should be 0-500 typically)
            original_rows = len(df)
            df = df[(df['AQI'] >= 0) & (df['AQI'] <= 500)]
            removed_rows = original_rows - len(df)
            if removed_rows > 0:
                print(f"  Removed {removed_rows} rows with unrealistic AQI values")
            
            print(f"AQI statistics after cleaning:")
            print(f"  Min AQI: {df['AQI'].min()}")
            print(f"  Max AQI: {df['AQI'].max()}")
            print(f"  Mean AQI: {df['AQI'].mean():.2f}")
        
        print(f"Data shape: {df.shape}")
        print(f"Selected features ({len(available_features)}): {available_features}")
        
        return df
    
    def create_aqi_categories(self, df):
        """Create AQI categories for classification"""
        def categorize_aqi(aqi):
            if aqi <= 50:
                return "Good"
            elif aqi <= 100:
                return "Satisfactory"
            elif aqi <= 200:
                return "Moderate"
            elif aqi <= 300:
                return "Poor"
            else:
                return "Very Poor"
        
        if 'AQI' in df.columns:
            df['AQI_Category'] = df['AQI'].apply(categorize_aqi)
            
            # Create label encoder for categories
            le = LabelEncoder()
            df['AQI_Category_Encoded'] = le.fit_transform(df['AQI_Category'])
            self.scalers['label_encoder'] = le
            
        return df
    
    def train_regression_models(self, df):
        """Train regression models to predict AQI value"""
        if 'AQI' not in df.columns:
            print("AQI column not found. Skipping regression models.")
            return
        
        print("\nTraining regression models...")
        
        X = df[self.feature_columns]
        y = df['AQI']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        lr_score = r2_score(y_test, lr_pred)
        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
        
        self.models['linear_regression'] = lr_model
        self.metadata['linear_regression'] = {
            'type': 'regression',
            'target': 'AQI',
            'r2_score': lr_score,
            'rmse': lr_rmse,
            'features': self.feature_columns
        }
        
        print(f"Linear Regression - R²: {lr_score:.4f}, RMSE: {lr_rmse:.4f}")
        
        # Decision Tree Regression
        dt_reg = DecisionTreeRegressor(random_state=42, max_depth=10)
        dt_reg.fit(X_train, y_train)
        dt_pred = dt_reg.predict(X_test)
        dt_score = r2_score(y_test, dt_pred)
        dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))
        
        self.models['decision_tree_regression'] = dt_reg
        self.metadata['decision_tree_regression'] = {
            'type': 'regression',
            'target': 'AQI',
            'r2_score': dt_score,
            'rmse': dt_rmse,
            'features': self.feature_columns
        }
        
        print(f"Decision Tree Regression - R²: {dt_score:.4f}, RMSE: {dt_rmse:.4f}")
        
        # Random Forest Regression
        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf_reg.fit(X_train, y_train)
        rf_pred = rf_reg.predict(X_test)
        rf_score = r2_score(y_test, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        
        self.models['random_forest_regression'] = rf_reg
        self.metadata['random_forest_regression'] = {
            'type': 'regression',
            'target': 'AQI',
            'r2_score': rf_score,
            'rmse': rf_rmse,
            'features': self.feature_columns
        }
        
        print(f"Random Forest Regression - R²: {rf_score:.4f}, RMSE: {rf_rmse:.4f}")
        
        # Show feature importance for Random Forest (most reliable indicator)
        feature_importance = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': rf_reg.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature Importance (Random Forest Regression):")
        for _, row in feature_importance.head().iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    def train_classification_models(self, df):
        """Train classification models to predict AQI category"""
        if 'AQI_Category' not in df.columns:
            print("AQI_Category column not found. Skipping classification models.")
            return
        
        print("\nTraining classification models...")
        
        X = df[self.feature_columns]
        y = df['AQI_Category']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Decision Tree Classification
        dt_clf = DecisionTreeClassifier(random_state=42, max_depth=10)
        dt_clf.fit(X_train, y_train)
        dt_pred = dt_clf.predict(X_test)
        dt_acc = accuracy_score(y_test, dt_pred)
        
        self.models['decision_tree_classification'] = dt_clf
        self.metadata['decision_tree_classification'] = {
            'type': 'classification',
            'target': 'AQI_Category',
            'accuracy': dt_acc,
            'features': self.feature_columns,
            'classes': list(dt_clf.classes_)
        }
        
        print(f"Decision Tree Classification - Accuracy: {dt_acc:.4f}")
        
        # Random Forest Classification
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf_clf.fit(X_train, y_train)
        rf_pred = rf_clf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)
        
        self.models['random_forest_classification'] = rf_clf
        self.metadata['random_forest_classification'] = {
            'type': 'classification',
            'target': 'AQI_Category',
            'accuracy': rf_acc,
            'features': self.feature_columns,
            'classes': list(rf_clf.classes_)
        }
        
        print(f"Random Forest Classification - Accuracy: {rf_acc:.4f}")
    
    def save_models(self):
        """Save all trained models and metadata"""
        print("\nSaving models...")
        
        # Save individual models
        for model_name, model in self.models.items():
            model_path = f'saved_models/{model_name}.pkl'
            joblib.dump(model, model_path)
            print(f"Saved: {model_path}")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_path = f'saved_models/{scaler_name}.pkl'
            joblib.dump(scaler, scaler_path)
            print(f"Saved: {scaler_path}")
        
        # Save metadata
        metadata_with_timestamp = {
            'timestamp': datetime.now().isoformat(),
            'feature_columns': self.feature_columns,
            'models': self.metadata
        }
        
        with open('saved_models/model_metadata.json', 'w') as f:
            json.dump(metadata_with_timestamp, f, indent=4)
        
        print("Saved: saved_models/model_metadata.json")
        print(f"\nAll models saved successfully! Total models: {len(self.models)}")
    
    def train_and_save_all(self):
        """Complete pipeline to train and save all models"""
        print("Starting AQI Model Training Pipeline...")
        print("=" * 50)
        
        # Load and preprocess data
        df = self.load_and_preprocess_data()
        
        # Create AQI categories
        df = self.create_aqi_categories(df)
        
        # Train models
        self.train_regression_models(df)
        self.train_classification_models(df)
        
        # Save everything
        self.save_models()
        
        print("\n" + "=" * 50)
        print("Model Training Pipeline Complete!")
        print("=" * 50)


if __name__ == "__main__":
    # Run the complete pipeline
    model_saver = AQIModelSaver()
    model_saver.train_and_save_all()