# feature_pipeline/feature_engineer.py

import pandas as pd
import os
from glob import glob

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates time-based features from the datetime column.
    """
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    print("Time-based features created: 'hour', 'dayofweek', 'month'")
    return df

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates derived features like AQI change rate.
    """
    # Sort by datetime to ensure correct calculation
    df = df.sort_values(by='datetime')
    # Calculate the change in AQI from the previous hour
    df['aqi_change_rate'] = df['aqi'].diff()
    # The first row will have NaN, we can fill it with 0
    df['aqi_change_rate'].fillna(0, inplace=True)
    print("Derived feature created: 'aqi_change_rate'")
    return df

def find_latest_csv_path(folder_path: str) -> str | None:
    """
    Finds the most recently created CSV file in a folder.
    """
    list_of_files = glob(os.path.join(folder_path, '*.csv'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def run_feature_engineering(raw_data_path: str) -> pd.DataFrame:
    """
    Main function to run the full feature engineering pipeline.
    """
    print(f"--- Starting Feature Engineering for {raw_data_path} ---")
    
    # Load the raw data
    try:
        df = pd.read_csv(raw_data_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        print("Raw data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The file {raw_data_path} was not found.")
        return pd.DataFrame()

    # Create features
    df = create_time_features(df)
    df = create_derived_features(df)
    
    # Reorder columns for better readability
    feature_cols = ['datetime', 'city', 'aqi', 'hour', 'dayofweek', 'month', 'aqi_change_rate', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    df = df[feature_cols]
    
    print("\n--- Feature Engineered Data Sample ---")
    print(df.head())
    
    return df

# --- This part runs when you execute the script directly ---
if __name__ == "__main__":
    # Define paths
    raw_data_folder = '../data/raw'  # <-- YEH PATH SAHI HAI
    processed_data_folder = '../data/processed' # <-- YEH BHI SAHI HAI
    
    # Create the processed data folder if it doesn't exist
    os.makedirs(processed_data_folder, exist_ok=True)
    
    # Find the latest raw data file
    latest_raw_file = find_latest_csv_path(raw_data_folder)
    
    if latest_raw_file:
        # Run the engineering process
        processed_df = run_feature_engineering(latest_raw_file)
        
        if not processed_df.empty:
            # Save the processed data
            base_filename = os.path.basename(latest_raw_file)
            processed_filename = base_filename.replace('raw', 'processed')
            output_path = os.path.join(processed_data_folder, processed_filename)
            processed_df.to_csv(output_path, index=False)
            print(f"\nSuccessfully saved processed data to: {output_path}")
    else:
        print(f"No raw data files found in '{raw_data_folder}'. Please run data_fetcher.py first.")
