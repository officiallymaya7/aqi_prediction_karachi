# 🌫️ AQI Prediction System for Karachi

An **end-to-end AI-driven Air Quality Index (AQI) Prediction System** built for Karachi.  
This project integrates **OpenWeatherMap API**, **Hopsworks Feature Store API**, and **GitHub Actions CI/CD** to automatically collect, process, and predict air quality in real time.

---

## 🧠 Overview

The system predicts Karachi’s **Air Quality Index (AQI)** based on live environmental data.  
It fetches pollutant readings via **OpenWeatherMap API**, processes them through feature engineering pipelines, stores them in **Hopsworks**, and retrains models automatically using **GitHub Actions**.  
Predictions are visualized using a **Streamlit web app** for easy interpretation.

---

## 🎯 Objectives

- Automate AQI data collection and preprocessing  
- Build a robust ML model for hourly AQI prediction  
- Manage features using Hopsworks API  
- Implement continuous model retraining and deployment through CI/CD  
- Provide real-time visualization through Streamlit  

---

## 📂 Project Structure
├── data/
│ ├── raw/ # Raw API data
│ └── processed/ # Cleaned & feature-engineered data
├── feature_pipeline/
│ ├── data_fetcher.py # Fetch data from OpenWeatherMap
│ ├── feature_engineer.py # Generate derived/time-based features
│ └── feature_store.py # Store processed data using Hopsworks API
├── training_pipeline/
│ └── model_trainer.py # Train and evaluate ML models
├── prediction_service/
│ └── app.py # Streamlit app for real-time AQI prediction
├── workflows/
│ └── github_action.yml # Automated CI/CD workflow
├── notebooks/
│ └── eda.ipynb # Exploratory Data Analysis
├── utils/
│ └── config.py # API keys, paths, and constants
├── requirements.txt
└── README.md


---

## 🌐 Data Collection

Data is obtained through the **OpenWeatherMap Air Pollution API** for Karachi’s coordinates.  
The pipeline automates hourly collection, converting JSON responses into structured tabular form.

**Collected Attributes:**  
`datetime`, `aqi`, `co`, `no`, `no2`, `o3`, `so2`, `pm2_5`, `pm10`, `nh3`

**Script:** `feature_pipeline/data_fetcher.py`

**Steps:**
1. API call using city coordinates  
2. Extract pollutants and timestamp  
3. Store as CSV in `/data/raw/`

---

## 📊 Exploratory Data Analysis (EDA)

Performed in `notebooks/eda.ipynb`.

**Findings:**
- Strong positive correlation between **PM2.5** / **PM10** and AQI.  
- AQI values fluctuate with time (especially during traffic hours).  
- Seasonal variations indicate higher pollution in winter months.  

Visualizations include:
- Correlation heatmaps  
- AQI trend plots  
- Hourly variation graphs  

---

## ⚙️ Feature Engineering

**Script:** `feature_pipeline/feature_engineer.py`

### 🔹 Created Features:
| Feature | Description |
|----------|--------------|
| `hour` | Hour of the day (0–23) |
| `dayofweek` | Day of the week (0=Mon) |
| `month` | Month extracted from timestamp |
| `aqi_change_rate` | Change in AQI compared to the previous hour |

### 🔹 Purpose:
To capture **temporal and trend-based dependencies**, improving prediction accuracy.

Processed data is saved in `/data/processed/`.

---

## 🧩 Feature Store Integration (Hopsworks API)

**Script:** `feature_pipeline/feature_store.py`

The processed features are stored and versioned in **Hopsworks Feature Store** using its Python API.

**Implementation:**
1. Connect to Hopsworks using `HOPSWORKS_API_KEY`  
2. Create Feature Group → `aqi_features_karachi`  
3. Insert processed pandas DataFrame  
4. Enable version control and feature retrieval for ML models  

This ensures consistency between training and prediction environments.

---

## 🤖 Model Training & Evaluation

**Script:** `training_pipeline/model_trainer.py`

Three models were trained and evaluated using MSE, MAE, and R² metrics.

| Model | RMSE | MAE | R² |
|--------|------|------|------|
| Linear Regression | 0.45 | 0.32 | 0.75 |
| Random Forest | 0.22 | 0.08 | 0.91 |
| XGBoost | **0.19** | **0.05** | **0.94** |

🏆 **Best Model:** XGBoost Regressor  
Saved as `best_aqi_predictor.pkl`

The model achieves **94% variance explanation**, indicating strong performance.

---

## 🔄 Continuous Integration & Deployment (CI/CD)

**File:** `workflows/github_action.yml`

This automation pipeline ensures that every time new data is fetched, the model retrains and redeploys automatically.

### 🔁 Workflow Steps:
1. **Checkout Code** → Pulls latest repo  
2. **Set up Python 3.11** → Ensures consistency  
3. **Install Dependencies** → From `requirements.txt`  
4. **Run Feature Pipeline**
   - `data_fetcher.py`
   - `feature_engineer.py`
   - `feature_store.py`  
5. **Train Model** → Runs `model_trainer.py`  
6. **Save Model** → Uploads latest model to repository or cloud storage  

🕐 **Runs Automatically Every Hour** via cron job (`schedule:` in YAML).  

---

## 🌍 Real-Time Prediction Dashboard

**Script:** `prediction_service/app.py`  
**Framework:** Streamlit  

### 💡 Features:
- Displays **latest predicted AQI**
- Graphs for pollutant-level contribution
- Trend visualization of previous hours
- User-friendly, minimal UI  

**Run Locally:**
```bash
streamlit run prediction_service/app.py


