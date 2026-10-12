# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
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
    "client_kwargs": {"endpoint_url": MINIO_ENDPOINT},
}

TRAINING_PATH = "s3://uber-data/demand/gnn_training"
NODES_PATH = "s3://uber-data/gnn/nodes"
EDGES_PATH = "s3://uber-data/gnn/edges"
MODEL_PATH = "/app/models/demand_gcn.pt"


class DemandGCN(nn.Module):
    def __init__(self, hidden_channels=32):
        super().__init__()
        self.conv1 = GCNConv(3, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.output_layer = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, data.edge_index))
        return self.output_layer(x).squeeze(-1)


def main():
    print("Loading data...")
    demand = pd.read_parquet(
        TRAINING_PATH, storage_options=STORAGE_OPTIONS
    )
    nodes = pd.read_parquet(
        NODES_PATH, storage_options=STORAGE_OPTIONS
    ).sort_values("node_index").reset_index(drop=True)
    edges = pd.read_parquet(
        EDGES_PATH, storage_options=STORAGE_OPTIONS
    )

    demand["pickup_hour"] = pd.to_datetime(demand["pickup_hour"])
    demand["target_hour"] = pd.to_datetime(demand["target_hour"])

    checkpoint = torch.load(
        MODEL_PATH, map_location="cpu", weights_only=False
    )

    model = DemandGCN(checkpoint["hidden_channels"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    zone_to_index = {
        int(row.location_id): int(row.node_index)
        for row in nodes.itertuples(index=False)
    }

    edge_index = torch.tensor(
        edges[["source_index", "target_index"]].to_numpy().T.copy(),
        dtype=torch.long,
    )
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    graph = Data(edge_index=edge_index, num_nodes=len(nodes))

    hours = sorted(demand["pickup_hour"].dropna().unique())
    val_end = int(len(hours) * 0.85)
    test_hours = hours[val_end:]

    records = []

    for hour in test_hours:
        rows = demand[demand["pickup_hour"] == hour].copy()
        if rows.empty:
            continue

        features = np.zeros((len(nodes), 3), dtype=np.float32)
        actual_by_zone = {}

        for row in rows.itertuples(index=False):
            zone = int(row.pickup_location)
            if zone not in zone_to_index:
                continue

            idx = zone_to_index[zone]
            hod = float(row.hour_of_day)

            features[idx] = [
                (float(row.trip_count) - checkpoint["demand_mean"])
                / checkpoint["demand_std"],
                np.sin(2 * np.pi * hod / 24),
                np.cos(2 * np.pi * hod / 24),
            ]

            actual_by_zone[zone] = float(row.target_demand)

        graph.x = torch.tensor(features, dtype=torch.float32)

        with torch.no_grad():
            normalized = model(graph).numpy()

        forecasts = np.maximum(
            normalized * checkpoint["target_std"]
            + checkpoint["target_mean"],
            0,
        )

        for zone, actual in actual_by_zone.items():
            idx = zone_to_index[zone]
            row = rows[rows["pickup_location"] == zone].iloc[0]

            records.append({
                "input_hour": hour,
                "target_hour": row["target_hour"],
                "pickup_location": zone,
                "actual_demand": actual,
                "previous_hour_prediction": float(row["trip_count"]),
                "gnn_prediction": float(forecasts[idx]),
            })

    comparison = pd.DataFrame(records)

    if comparison.empty:
        raise ValueError("No test records available.")

    # Verify next-hour alignment.
    offset = (
        comparison["target_hour"] - comparison["input_hour"]
    ).dt.total_seconds() / 3600

    print("Test input hours:", comparison["input_hour"].nunique())
    print("Zone-hour records:", len(comparison))
    print("One-hour aligned records:", int((offset == 1).sum()))

    comparison = comparison[offset == 1].dropna().copy()

    if comparison.empty:
        raise ValueError("No correctly aligned one-hour records.")

    def metrics(actual, predicted):
        errors = predicted - actual
        return (
            float(np.mean(np.abs(errors))),
            float(np.sqrt(np.mean(errors ** 2))),
        )

    actual = comparison["actual_demand"].to_numpy(dtype=float)

    baseline_mae, baseline_rmse = metrics(
        actual,
        comparison["previous_hour_prediction"].to_numpy(dtype=float),
    )
    gnn_mae, gnn_rmse = metrics(
        actual,
        comparison["gnn_prediction"].to_numpy(dtype=float),
    )

    print("\nFair comparison on identical records")
    print("-------------------------------------")
    print(f"Previous-hour baseline MAE:  {baseline_mae:.4f}")
    print(f"Previous-hour baseline RMSE: {baseline_rmse:.4f}")
    print(f"GNN MAE:                     {gnn_mae:.4f}")
    print(f"GNN RMSE:                    {gnn_rmse:.4f}")

    comparison.to_csv("model_comparison.csv", index=False)
    print("\nSaved: model_comparison.csv")


if __name__ == "__main__":
    main()