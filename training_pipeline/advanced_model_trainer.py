# training_pipeline/advanced_model_trainer.py

import pandas as pd
import numpy as np
import os
import hopsworks
import joblib
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor # Make sure to install xgboost: pip install xgboost
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load environment variables
load_dotenv()

def connect_to_hopsworks():
    """Connects to the Hopsworks project."""
    try:
        project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
        print("Successfully connected to Hopsworks.")
        return project
    except Exception as e:
        print(f"Error connecting to Hopsworks: {e}")
        return None

def fetch_features_from_store(project):
    """Fetches features from the Hopsworks feature store."""
    try:
        fs = project.get_feature_store()
        feature_group_name = os.getenv("FEATURE_GROUP_NAME")
        feature_group = fs.get_feature_group(name=feature_group_name, version=1)
        
        print(f"Fetching features from '{feature_group_name}'...")
        feature_df = feature_group.read()
        print("Features fetched successfully.")
        return feature_df
    except Exception as e:
        print(f"Error fetching features: {e}")
        return None

def prepare_data_for_training(df):
    """Prepares the data for model training."""
    df = df.drop(columns=['datetime', 'city'])
    X = df.drop('aqi', axis=1)
    y = df['aqi']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data prepared. Training set size: {X_train.shape}, Testing set size: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def run_model_experiments(X_train, X_test, y_train, y_test):
    """Trains multiple models and compares their performance."""
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, objective='reg:squarederror')
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
        print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
        
    return results, models

def find_and_save_best_model(results, models):
    """Finds the best model based on R2 score and saves it."""
    best_model_name = max(results, key=lambda k: results[k]['R2'])
    best_model = models[best_model_name]
    
    print(f"\n🏆 Best model is {best_model_name} with R² score of {results[best_model_name]['R2']:.2f}")
    
    model_path = '../models/best_aqi_predictor.pkl'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"Best model saved to: {model_path}")

# --- Main execution ---
if __name__ == "__main__":
    project = connect_to_hopsworks()
    if not project: exit()

    features_df = fetch_features_from_store(project)
    if features_df is None: exit()

    X_train, X_test, y_train, y_test = prepare_data_for_training(features_df)
    
    results, trained_models = run_model_experiments(X_train, X_test, y_train, y_test)
    
    find_and_save_best_model(results, trained_models)
    