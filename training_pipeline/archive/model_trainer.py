# training_pipeline/model_trainer.py

import pandas as pd
import numpy as np
import os
import hopsworks
import joblib
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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
    # Drop columns that are not features
    df = df.drop(columns=['datetime', 'city'])
    
    # Define features (X) and target (y)
    X = df.drop('aqi', axis=1)
    y = df['aqi']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Data prepared. Training set size: {X_train.shape}, Testing set size: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """Trains a RandomForest model and evaluates it."""
    print("\n--- Training RandomForest Model ---")
    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Train the model
    model.fit(X_train, y_train)
    print("Model training completed.")
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Model Evaluation ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R²): {r2:.2f}")
    
    return model

def save_model(model, model_path='../models/aqi_predictor_karachi.pkl'):
    """Saves the trained model to a file."""
    # Create the models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nModel saved successfully to: {model_path}")

# --- This part runs when you execute the script directly ---
if __name__ == "__main__":
    # Connect to Hopsworks
    project = connect_to_hopsworks()
    if not project:
        exit()

    # Fetch the features
    features_df = fetch_features_from_store(project)
    
    if features_df is not None:
        # Prepare data for training
        X_train, X_test, y_train, y_test = prepare_data_for_training(features_df)
        
        # Train and evaluate the model
        trained_model = train_and_evaluate_model(X_train, X_test, y_train, y_test)
        
        # Save the trained model
        save_model(trained_model)
    else:
        print("\n--- Failed to fetch features. ---")
