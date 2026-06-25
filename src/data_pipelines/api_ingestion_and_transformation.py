# Databricks / Synapse PySpark Notebook
# Title: JFKIAT API Ingestion & Transformation (Bronze to Silver)

import requests
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, from_json, current_timestamp, lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType

# Initialize Spark (Handled automatically in Synapse/Databricks, included for local context)
spark = SparkSession.builder.appName("JFKIAT_Operations_ETL").getOrCreate()

# ==========================================
# 1. INGEST SAMSARA API: Flight Movements
# Tracking aircraft from runway to hardstand to terminal
# ==========================================
print("Ingesting Samsara Flight Movement Data...")

# Mocking API Call (Replace with actual requests.get() and Secret Scopes for auth)
# response = requests.get("https://api.samsara.com/fleet/vehicles/locations", headers={"Authorization": f"Bearer {TOKEN}"})
samsara_mock_data = [
    {"flight_id": "DL405", "vehicle_id": "TUG_001", "status": "Runway", "timestamp": "2026-06-25T08:00:00Z"},
    {"flight_id": "DL405", "vehicle_id": "TUG_001", "status": "Hardstand", "timestamp": "2026-06-25T08:15:00Z"},
    {"flight_id": "DL405", "vehicle_id": "TUG_001", "status": "Terminal_Gate_B", "timestamp": "2026-06-25T08:30:00Z"}
]

# Create DataFrame
flight_movement_df = spark.createDataFrame(samsara_mock_data)

# Transformation: Cast timestamps and add ingestion metadata
silver_flight_df = flight_movement_df \
    .withColumn("event_timestamp", col("timestamp").cast(TimestampType())) \
    .withColumn("ingestion_timestamp", current_timestamp()) \
    .drop("timestamp")

# Write to Silver Data Lake / Delta Table
# silver_flight_df.write.format("delta").mode("append").saveAsTable("Silver_Flight_Movements")


# ==========================================
# 2. INGEST VENDOR APP: Parking & Courier Logistics
# ==========================================
print("Ingesting Vendor Parking Booking Data...")

vendor_mock_data = [
    {"booking_id": "B_9921", "vendor": "FedEx", "lot_id": "Cargo_Lot_A", "entry_time": "2026-06-25T07:30:00Z", "status": "Dropped_Goods"},
    {"booking_id": "B_9922", "vendor": "DHL", "lot_id": "Cargo_Lot_B", "entry_time": "2026-06-25T07:45:00Z", "status": "Pickup_Courier"}
]

vendor_df = spark.createDataFrame(vendor_mock_data)
silver_vendor_df = vendor_df \
    .withColumn("entry_time", col("entry_time").cast(TimestampType())) \
    .withColumn("ingestion_timestamp", current_timestamp())

# Write to Silver Data Lake / Delta Table
# silver_vendor_df.write.format("delta").mode("append").saveAsTable("Silver_Vendor_Logistics")


# ==========================================
# 3. INGEST SECURITY API: SOC, CBP, and Boarding
# ==========================================
print("Ingesting Airport Security & CBP Checkpoint Data...")

security_mock_data = [
    {"terminal": "T4", "checkpoint": "CBP_Primary", "wait_time_minutes": 14, "passenger_throughput": 450, "timestamp": "2026-06-25T09:00:00Z"},
    {"terminal": "T4", "checkpoint": "SOC_TSA_PreCheck", "wait_time_minutes": 5, "passenger_throughput": 800, "timestamp": "2026-06-25T09:00:00Z"}
]

security_df = spark.createDataFrame(security_mock_data)
silver_security_df = security_df \
    .withColumn("recorded_timestamp", col("timestamp").cast(TimestampType())) \
    .withColumn("ingestion_timestamp", current_timestamp()) \
    .drop("timestamp")

# Write to Silver Data Lake / Delta Table
# silver_security_df.write.format("delta").mode("append").saveAsTable("Silver_Security_Metrics")

print("Bronze to Silver Pipeline Complete.")
