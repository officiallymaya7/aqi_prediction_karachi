# feature_pipeline/feature_store.py

import pandas as pd
import os
import hopsworks
# from dotenv import load_dotenv

# Load environment variables from .env file
# load_dotenv()

def connect_to_hopsworks():
    """Connects to the Hopsworks project."""
    try:
        project = hopsworks.login(
            api_key_value=os.getenv("HOPSWORKS_API_KEY"),
            # project_name=os.getenv("HOPSWORKS_PROJECT_NAME")
        )
        print("Successfully connected to Hopsworks.")
        return project
    except Exception as e:
        print(f"Error connecting to Hopsworks: {e}")
        return None

def create_or_get_feature_group(project, feature_df):
    """Creates a feature group in Hopsworks or gets it if it exists."""
    try:
        # Get the feature store handle
        fs = project.get_feature_store()
        
        # Define feature group metadata
        feature_group_name = os.getenv("FEATURE_GROUP_NAME")
        feature_group_version = 1
        
        # Create or get the feature group
        feature_group = fs.get_or_create_feature_group(
            name=feature_group_name,
            version=feature_group_version,
            description="AQI and weather features for Karachi",
            primary_key=['datetime'], # Each record is unique by its datetime
            event_time='datetime',
            online_enabled=False, # We are not using online feature store for this project
        )
        print(f"Feature group '{feature_group_name}' created or retrieved successfully.")
        return feature_group
    except Exception as e:
        print(f"Error creating/getting feature group: {e}")
        return None

def insert_data_into_feature_group(feature_group, df):
    """Inserts the DataFrame into the Hopsworks feature group."""
    try:
        # Insert the data into the feature group
        feature_group.insert(df)
        print("Data successfully inserted into the feature group.")
    except Exception as e:
        print(f"Error inserting data into feature group: {e}")

def run_feature_store_pipeline():
    """Main function to run the feature store pipeline."""
    # Load the processed data
    # NOTE: This script assumes the processed file has a specific name.
    # For a more robust solution, you might want to find the latest file like in feature_engineer.py
    processed_data_path = '../data/processed/karachi_aqi_20251104_to_20251107.csv'
    
    try:
        processed_df = pd.read_csv(processed_data_path)
        processed_df['datetime'] = pd.to_datetime(processed_df['datetime'])
        print("Processed data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {processed_data_path}")
        print("Please make sure you have run feature_engineer.py first.")
        return

    # Connect to Hopsworks
    project = connect_to_hopsworks()
    if not project:
        return

    # Create or get the feature group
    feature_group = create_or_get_feature_group(project, processed_df)
    if not feature_group:
        return

    # Insert data into the feature group
    insert_data_into_feature_group(feature_group, processed_df)

# --- This part runs when you execute the script directly ---
if __name__ == "__main__":
    run_feature_store_pipeline()