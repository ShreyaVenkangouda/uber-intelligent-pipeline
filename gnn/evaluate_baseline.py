import os
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd

MINIO_ENDPOINT = "http://minio:9000"

STORAGE_OPTIONS = {
    "key": "admin",
    "secret": "admin12345",
    "client_kwargs": {
        "endpoint_url": MINIO_ENDPOINT,
    },
}

TRAINING_PATH = "s3://uber-data/demand/gnn_training"
OUTPUT_DIR = "/app/evaluation"


def main():
    print("Loading demand dataset...")

    demand = pd.read_parquet(
        TRAINING_PATH,
        storage_options=STORAGE_OPTIONS,
    )

    demand["pickup_hour"] = pd.to_datetime(demand["pickup_hour"])
    demand["target_hour"] = pd.to_datetime(demand["target_hour"])

    required = {
        "pickup_hour",
        "target_hour",
        "pickup_location",
        "trip_count",
        "target_demand",
    }

    missing = required - set(demand.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Use the same chronological test-hour selection as training.
    hours = sorted(demand["pickup_hour"].dropna().unique())

    if len(hours) < 10:
        raise ValueError("Not enough hourly observations.")

    train_end = int(len(hours) * 0.70)
    val_end = int(len(hours) * 0.85)
    test_hours = hours[val_end:]

    test = demand[
        demand["pickup_hour"].isin(test_hours)
    ].copy()

    # The previous-hour baseline predicts that next-hour demand
    # will equal the demand observed in the input hour.
    test["baseline_prediction"] = test["trip_count"].astype(float)
    test["actual_demand"] = test["target_demand"].astype(float)

    # Verify that each label refers to exactly the next hour.
    test["target_offset_hours"] = (
        test["target_hour"] - test["pickup_hour"]
    ).dt.total_seconds() / 3600

    valid_alignment = test["target_offset_hours"] == 1

    print("Total test records:", len(test))
    print("Records aligned to exactly one hour:", int(valid_alignment.sum()))
    print("Records with unexpected target offset:", int((~valid_alignment).sum()))

    test = test[valid_alignment].dropna(
        subset=["baseline_prediction", "actual_demand"]
    ).copy()

    if test.empty:
        raise ValueError(
            "No valid test records found. Check target_hour alignment."
        )

    errors = (
        test["baseline_prediction"] - test["actual_demand"]
    )

    mae = errors.abs().mean()
    rmse = np.sqrt((errors ** 2).mean())

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    test["error"] = errors
    test["absolute_error"] = errors.abs()

    output_path = os.path.join(
        OUTPUT_DIR,
        "baseline_predictions.csv",
    )

    test[
        [
            "pickup_hour",
            "target_hour",
            "pickup_location",
            "trip_count",
            "actual_demand",
            "baseline_prediction",
            "error",
            "absolute_error",
        ]
    ].to_csv(output_path, index=False)

    print("\nPrevious-hour baseline results")
    print("------------------------------")
    print("Test input hours:", len(test_hours))
    print("Zone-hour records evaluated:", len(test))
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()