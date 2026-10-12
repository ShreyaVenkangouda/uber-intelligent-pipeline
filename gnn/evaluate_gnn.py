import os
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn.functional as F

# pyrefly: ignore [missing-import]
from torch import nn
# pyrefly: ignore [missing-import]
from torch_geometric.data import Data
# pyrefly: ignore [missing-import]
from torch_geometric.nn import GCNConv


MINIO_ENDPOINT = "http://minio:9000"

STORAGE_OPTIONS = {
    "key": "admin",
    "secret": "admin12345",
    "client_kwargs": {
        "endpoint_url": MINIO_ENDPOINT,
    },
}

TRAINING_PATH = "s3://uber-data/demand/gnn_training"
NODES_PATH = "s3://uber-data/gnn/nodes"
EDGES_PATH = "s3://uber-data/gnn/edges"

MODEL_PATH = "/app/models/demand_gcn.pt"
OUTPUT_DIR = "/app/evaluation"

DEVICE = torch.device("cpu")


class DemandGCN(nn.Module):
    def __init__(self, hidden_channels=32):
        super().__init__()
        self.conv1 = GCNConv(3, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.output_layer = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, data.edge_index)
        x = F.relu(x)
        return self.output_layer(x).squeeze(-1)


def main():
    print("Loading datasets...")

    demand = pd.read_parquet(
        TRAINING_PATH,
        storage_options=STORAGE_OPTIONS,
    )

    nodes = pd.read_parquet(
        NODES_PATH,
        storage_options=STORAGE_OPTIONS,
    ).sort_values("node_index").reset_index(drop=True)

    edges = pd.read_parquet(
        EDGES_PATH,
        storage_options=STORAGE_OPTIONS,
    )

    demand["pickup_hour"] = pd.to_datetime(demand["pickup_hour"])
    demand["target_hour"] = pd.to_datetime(demand["target_hour"])

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model = DemandGCN(
        hidden_channels=checkpoint["hidden_channels"]
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    demand_mean = float(checkpoint["demand_mean"])
    demand_std = float(checkpoint["demand_std"])
    target_mean = float(checkpoint["target_mean"])
    target_std = float(checkpoint["target_std"])

    if demand_std <= 0 or target_std <= 0:
        raise ValueError("Invalid normalization standard deviation.")

    zone_to_index = {
        int(row.location_id): int(row.node_index)
        for row in nodes.itertuples(index=False)
    }

    edge_index = torch.tensor(
        edges[["source_index", "target_index"]].to_numpy().T.copy(),
        dtype=torch.long,
    )

    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    graph = Data(
        edge_index=edge_index.to(DEVICE),
        num_nodes=len(nodes),
    )

    # Use the same chronological 70/15/15 split as training.
    hours = sorted(demand["pickup_hour"].dropna().unique())

    if len(hours) < 10:
        raise ValueError("Not enough hourly observations.")

    train_end = int(len(hours) * 0.70)
    val_end = int(len(hours) * 0.85)
    test_hours = hours[val_end:]

    print("Total input hours:", len(hours))
    print("Historical test hours:", len(test_hours))

    records = []

    for input_hour in test_hours:
        rows = demand[demand["pickup_hour"] == input_hour].copy()

        if rows.empty:
            continue

        # Use only features available at this input hour.
        features = np.zeros((len(nodes), 3), dtype=np.float32)
        actual_by_zone = {}

        for row in rows.itertuples(index=False):
            zone = int(row.pickup_location)

            if zone not in zone_to_index:
                continue

            idx = zone_to_index[zone]
            hour_of_day = float(row.hour_of_day)

            features[idx] = [
                (float(row.trip_count) - demand_mean) / demand_std,
                np.sin(2 * np.pi * hour_of_day / 24),
                np.cos(2 * np.pi * hour_of_day / 24),
            ]

            # The dataset's target is the next-hour demand.
            # Keep only zones with labels from the prepared dataset.
            actual_by_zone[zone] = float(row.target_demand)

        graph.x = torch.tensor(features, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            normalized = model(graph).cpu().numpy()

        forecast = np.maximum(
            normalized * target_std + target_mean,
            0,
        )

        prediction_hour = input_hour + pd.Timedelta(hours=1)

        for zone, actual_value in actual_by_zone.items():
            idx = zone_to_index[zone]

            records.append({
                "input_hour": input_hour,
                "prediction_hour": prediction_hour,
                "pickup_location": zone,
                "actual_demand": actual_value,
                "predicted_demand": float(forecast[idx]),
            })

    if not records:
        raise ValueError("No historical test predictions were generated.")

    results = pd.DataFrame(records)
    results["error"] = (
        results["predicted_demand"] - results["actual_demand"]
    )
    results["absolute_error"] = results["error"].abs()
    results["squared_error"] = results["error"] ** 2

    # Overall metrics across test hours and zones.
    mae = results["absolute_error"].mean()
    rmse = np.sqrt(results["squared_error"].mean())

    # Metrics per forecast hour.
    hourly = (
        results.groupby("prediction_hour")
        .agg(
            zones_evaluated=("pickup_location", "nunique"),
            MAE=("absolute_error", "mean"),
            MSE=("squared_error", "mean"),
        )
        .reset_index()
    )
    hourly["RMSE"] = np.sqrt(hourly["MSE"])
    hourly = hourly.drop(columns=["MSE"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results.to_csv(
        os.path.join(OUTPUT_DIR, "historical_predictions.csv"),
        index=False,
    )

    hourly.to_csv(
        os.path.join(OUTPUT_DIR, "hourly_metrics.csv"),
        index=False,
    )

    print("\nHistorical evaluation results")
    print("-----------------------------")
    print("Forecast hours evaluated:", results["prediction_hour"].nunique())
    print("Zone-hour predictions:", len(results))
    print("Zones evaluated:", results["pickup_location"].nunique())
    print(f"Overall MAE:  {mae:.4f}")
    print(f"Overall RMSE: {rmse:.4f}")

    print("\nHourly metrics:")
    print(hourly.to_string(index=False))

    print("\nSaved results under:", OUTPUT_DIR)


if __name__ == "__main__":
    main()