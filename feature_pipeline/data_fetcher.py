# feature_pipeline/data_fetcher.py

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
# from dotenv import load_dotenv

# Load environment variables from .env file
# load_dotenv()

# --- Configuration ---
API_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_URL = "http://api.openweathermap.org/data/2.5"

def get_coordinates(city_name: str, api_key: str) -> tuple[float, float]:
    """
    Converts a city name to latitude and longitude using the OpenWeatherMap Geocoding API.
    """
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct"
    params = {
        'q': city_name,
        'limit': 1,
        'appid': api_key
    }
    try:
        response = requests.get(geo_url, params=params)
        response.raise_for_status()
        data = response.json()
        if not data:
            raise ValueError(f"City '{city_name}' not found.")
        lat = data[0]['lat']
        lon = data[0]['lon']
        print(f"Found coordinates for {city_name}: Lat={lat}, Lon={lon}")
        return lat, lon
    except requests.exceptions.RequestException as e:
        print(f"Error fetching coordinates: {e}")
        return None, None

def fetch_historical_aqi(city_name: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetches historical air quality data for a given city and date range.
    """
    if not API_KEY:
        print("Error: OPENWEATHER_API_KEY not found in environment variables.")
        return pd.DataFrame()

    lat, lon = get_coordinates(city_name, API_KEY)
    if lat is None or lon is None:
        return pd.DataFrame()

    all_data = []
    current_date = start_date
    
    while current_date <= end_date:
        start_ts = int(current_date.timestamp())
        end_ts = int((current_date + timedelta(days=1)).timestamp())
        
        pollution_url = f"{BASE_URL}/air_pollution/history"
        params = {
            'lat': lat,
            'lon': lon,
            'start': start_ts,
            'end': end_ts,
            'appid': API_KEY
        }
        
        try:
            print(f"Fetching data for {current_date.strftime('%Y-%m-%d')}...")
            response = requests.get(pollution_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if 'list' in data and data['list']:
                for item in data['list']:
                    record = {
                        'datetime': datetime.fromtimestamp(item['dt']),
                        'aqi': item['main']['aqi'],
                        'co': item['components']['co'],
                        'no': item['components']['no'],
                        'no2': item['components']['no2'],
                        'o3': item['components']['o3'],
                        'so2': item['components']['so2'],
                        'pm2_5': item['components']['pm2_5'],
                        'pm10': item['components']['pm10'],
                        'nh3': item['components']['nh3'],
                        'city': city_name
                    }
                    all_data.append(record)
            else:
                print(f"No data returned for {current_date.strftime('%Y-%m-%d')}")

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {current_date.strftime('%Y-%m-%d')}: {e}")
            
        current_date += timedelta(days=1)

    if not all_data:
        print("No data was fetched for the entire period.")
        return pd.DataFrame()
        
    df = pd.DataFrame(all_data)
    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
    df.sort_values('datetime', inplace=True)
    return df

# --- This part runs when you execute the script directly ---
if __name__ == "__main__":
    # Example: Fetch data for Karachi for the last 3 days
    CITY = "Karachi"  # <-- YEH Karachi KAR DIYA GAYA HAI
    END = datetime.now()
    START = END - timedelta(days=3)

    print(f"--- Starting Data Fetch for {CITY} from {START.strftime('%Y-%m-%d')} to {END.strftime('%Y-%m-%d')} ---")
    
    aqi_df = fetch_historical_aqi(CITY, START, END)
    
    if not aqi_df.empty:
        print("\n--- Fetched Data Sample ---")
        print(aqi_df.head())
        
        # Save the raw data to a CSV file for inspection
        os.makedirs('../data/raw', exist_ok=True)
        output_path = f"../data/raw/{CITY.lower()}_aqi_{START.strftime('%Y%m%d')}_to_{END.strftime('%Y%m%d')}.csv"
        aqi_df.to_csv(output_path, index=False)
        print(f"\nSuccessfully saved data to: {output_path}")
    else:
        print("\n--- Failed to fetch any data. Please check API key and city name. ---")