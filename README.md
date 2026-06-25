# jfk
# JFKIAT Daily Airport Operations & Forecasting

## Overview
This repository contains the data models and forecasting pipelines developed for daily operations at JFKIAT (JFK International Air Terminal). The project supports the Security Operations Center (SOC) and Customs and Border Protection (CBP) by providing hourly forecasts and operational insights to manage terminal flow efficiently.

## Key Objectives
* **Flight Operations:** Forecast and track daily flight arrivals and departures to anticipate terminal loads.
* **CBP & SOC Resource Allocation:** Model hourly passenger traffic to ensure optimal staffing and minimize wait times at security and customs checkpoints.
* **Traffic Management:** Forecast hourly vehicle and passenger traffic to optimize curbside drop-off, pick-up, and internal terminal flow.

## Architecture & Tech Stack
* **Forecasting Engine:** Python (`Prophet`, `pandas`) utilized to handle complex hourly seasonality, predicting passenger and vehicle volumes.
* **Data Modeling:** Separate relational data models structured for dimensional reporting.
* **Visualization & Reporting:** **Power BI** is used as the primary display layer. It consumes the structured data models to provide side-by-side variance analysis (Actuals vs. Forecast) for stakeholders.

---

## Forecasting Implementation

The following Python script demonstrates the core forecasting engine, which predicts future hourly traffic based on historical patterns using daily and weekly seasonality.

> **Note:** The Python script below is a simplified structural representation of the core forecasting logic. The actual production implementation scales to ingest massive volumes of historical operational data from the data warehouse (e.g., via SQL or PySpark pipelines) to train the models before outputting the true forecasts for Power BI consumption.

```python
import pandas as pd
from prophet import Prophet
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class JFKIATForecaster:
    def __init__(self, seasonality_prior_scale=10.0):
        """
        Initialize the forecasting model.
        Emphasizes daily and weekly seasonality for airport operations.
        """
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_prior_scale=seasonality_prior_scale
        )
        
    def prepare_data(self, df, datetime_col, metric_col):
        """
        Formats the dataframe to Prophet's required 'ds' (datetime) and 'y' (numeric) columns.
        """
        prepared_df = df.rename(columns={datetime_col: 'ds', metric_col: 'y'})
        prepared_df['ds'] = pd.to_datetime(prepared_df['ds'])
        return prepared_df

    def train(self, df):
        """Fits the historical data to the model."""
        logging.info("Training the forecasting model on historical data...")
        self.model.fit(df)
        logging.info("Model training complete.")

    def predict_future_hourly(self, hours_ahead=72):
        """Generates forecasts for the specified future hours."""
        logging.info(f"Generating hourly forecast for the next {hours_ahead} hours...")
        future_dates = self.model.make_future_dataframe(periods=hours_ahead, freq='H')
        forecast = self.model.predict(future_dates)
        
        # Return relevant output columns for the data model
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

if __name__ == "__main__":
    
    # ==========================================
    # 1. DATA INGESTION (Production vs. Sample)
    # ==========================================
    
    # PRODUCTION: Ingesting actual historical data from the Data Warehouse
    # import sqlalchemy
    # engine = sqlalchemy.create_engine('your_database_connection_string')
    # query = "SELECT timestamp, passenger_count FROM SOC_Historical_Traffic WHERE timestamp >= '2023-01-01'"
    # historical_df = pd.read_sql(query, engine)
    
    # SAMPLE: Mock structure for repository demonstration
    dates = pd.date_range(start='2026-06-01', periods=168, freq='H') 
    historical_df = pd.DataFrame({'timestamp': dates, 'passenger_count': 500}) 
    
    # ==========================================
    # 2. MODEL PIPELINE EXECUTION
    # ==========================================
    
    forecaster = JFKIATForecaster()
    train_df = forecaster.prepare_data(historical_df, datetime_col='timestamp', metric_col='passenger_count')
    
    # Train model on the historical data and predict next 24 hours
    forecaster.train(train_df)
    forecast_df = forecaster.predict_future_hourly(hours_ahead=24)
    
    # Output results for Power BI Data Model ingestion
    print("Forecast Output Sample:")
    print(forecast_df.tail())
    
    # Write back to target storage (e.g., SQL Fact Table or Delta Lake)
    # forecast_df.to_csv("hourly_forecast_output.csv", index=False)
