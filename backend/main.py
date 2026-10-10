import os

import pandas as pd
# pyrefly: ignore [missing-import]
import s3fs
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware

# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

app = FastAPI(
    title="UberPulse API",
    description="API for Uber trip analytics and GNN demand predictions",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "admin12345")

# pandas saved this as a single Parquet object without an extension.
PREDICTIONS_OBJECT = "uber-data/predictions/next_hour"


# --------------------------------------------------
# MINIO CONNECTION
# --------------------------------------------------

def get_s3_filesystem():
    return s3fs.S3FileSystem(
        key=MINIO_ACCESS_KEY,
        secret=MINIO_SECRET_KEY,
        client_kwargs={
            "endpoint_url": MINIO_ENDPOINT,
        },
        config_kwargs={
            "s3": {
                "addressing_style": "path",
            },
        },
    )


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "UberPulse API is running",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "UberPulse API",
    }


# --------------------------------------------------
# GNN PREDICTIONS
# --------------------------------------------------

@app.get("/api/predictions")
def get_predictions():
    try:
        fs = get_s3_filesystem()

        # Check the exact object written by predict_demand.py.
        if not fs.exists(PREDICTIONS_OBJECT):
            raise HTTPException(
                status_code=404,
                detail=(
                    "Prediction object not found in MinIO: "
                    + PREDICTIONS_OBJECT
                ),
            )

        # Read the Parquet file even though it has no .parquet extension.
        with fs.open(PREDICTIONS_OBJECT, "rb") as file:
            df = pd.read_parquet(file)

        required_columns = [
            "pickup_location",
            "node_index",
            "prediction_hour",
            "predicted_demand",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Unexpected prediction data schema",
                    "missing_columns": missing_columns,
                    "available_columns": list(df.columns),
                },
            )

        df["predicted_demand"] = pd.to_numeric(
            df["predicted_demand"],
            errors="coerce",
        )

        df = df.dropna(subset=["predicted_demand"])

        df = df.sort_values(
            "predicted_demand",
            ascending=False,
        ).reset_index(drop=True)

        prediction_hour = None

        if not df.empty:
            prediction_hour = str(
                pd.to_datetime(df["prediction_hour"]).max()
            )

        # Return only the top 50 zones to the dashboard.
        records_df = df.head(50).copy()

        # Ensure timestamps and NumPy values are JSON serializable.
        records_df = records_df.astype(object)
        records_df = records_df.where(pd.notna(records_df), None)

        records = records_df.to_dict(orient="records")

        for record in records:
            for key, value in record.items():
                if hasattr(value, "isoformat"):
                    record[key] = value.isoformat()
                elif hasattr(value, "item"):
                    record[key] = value.item()

        return {
            "count": int(len(df)),
            "returned": len(records),
            "prediction_hour": prediction_hour,
            "source": PREDICTIONS_OBJECT,
            "predictions": records,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not read predictions from MinIO: {str(exc)}",
        ) from exc

@app.get("/api/hourly-demand")
def get_hourly_demand():
    try:
        fs = get_s3_filesystem()
        dataset_path = "uber-data/demand/hourly"

        # The Spark job stores hourly demand as Parquet data.
        parquet_files = fs.glob(f"{dataset_path}/**/*.parquet")

        if not parquet_files:
            parquet_files = fs.glob(f"{dataset_path}/*.parquet")

        if not parquet_files:
            raise HTTPException(
                status_code=404,
                detail="Hourly demand Parquet files were not found in MinIO.",
            )

        frames = []
        for path in parquet_files:
            with fs.open(path, "rb") as file:
                frames.append(pd.read_parquet(file))

        df = pd.concat(frames, ignore_index=True)

        required = {"pickup_hour", "trip_count"}
        missing = required.difference(df.columns)
        if missing:
            raise HTTPException(
                status_code=500,
                detail=f"Missing hourly demand columns: {sorted(missing)}",
            )

        df["pickup_hour"] = pd.to_datetime(df["pickup_hour"])
        df["trip_count"] = pd.to_numeric(df["trip_count"], errors="coerce")
        df = df.dropna(subset=["pickup_hour", "trip_count"])

        # Sum across zones to get total trips per hour.
        hourly = (
            df.groupby("pickup_hour", as_index=False)["trip_count"]
            .sum()
            .sort_values("pickup_hour")
        )

        return {
            "count": len(hourly),
            "hourly_demand": [
                {
                    "hour": row.pickup_hour.isoformat(),
                    "trips": int(row.trip_count),
                }
                for row in hourly.itertuples(index=False)
            ],
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not read hourly demand from MinIO: {exc}",
        ) from exc