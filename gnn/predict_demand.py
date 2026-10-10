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

# =====================================================
# CONFIGURATION
# =====================================================

MINIO_ENDPOINT = "http://minio:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "admin12345"

TRAINING_PATH = "s3://uber-data/demand/gnn_training"
NODES_PATH = "s3://uber-data/gnn/nodes"
EDGES_PATH = "s3://uber-data/gnn/edges"
OUTPUT_PATH = "s3://uber-data/predictions/next_hour"

MODEL_PATH = "/app/models/demand_gcn.pt"

STORAGE_OPTIONS = {
    "key": MINIO_ACCESS_KEY,
    "secret": MINIO_SECRET_KEY,
    "client_kwargs": {
        "endpoint_url": MINIO_ENDPOINT
    }
}

DEVICE = torch.device("cpu")


# =====================================================
# MODEL DEFINITION — MUST MATCH TRAINING SCRIPT
# =====================================================

class DemandGCN(nn.Module):
    def __init__(self, hidden_channels=32):
        super().__init__()

        self.conv1 = GCNConv(3, hidden_channels)
        self.conv2 = GCNConv(
            hidden_channels,
            hidden_channels
        )
        self.output_layer = nn.Linear(
            hidden_channels,
            1
        )

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=0.2,
            training=self.training
        )
        x = self.conv2(x, data.edge_index)
        x = F.relu(x)

        return self.output_layer(x).squeeze(-1)


# =====================================================
# LOAD INPUT DATA
# =====================================================

print("Loading demand data and graph...")

training_df = pd.read_parquet(
    TRAINING_PATH,
    storage_options=STORAGE_OPTIONS
)

nodes_df = pd.read_parquet(
    NODES_PATH,
    storage_options=STORAGE_OPTIONS
)

edges_df = pd.read_parquet(
    EDGES_PATH,
    storage_options=STORAGE_OPTIONS
)

training_df["pickup_hour"] = pd.to_datetime(
    training_df["pickup_hour"]
)

nodes_df = nodes_df.sort_values(
    "node_index"
).reset_index(drop=True)

# =====================================================
# LOAD TRAINED MODEL
# =====================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model = DemandGCN(
    hidden_channels=checkpoint["hidden_channels"]
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

demand_mean = checkpoint["demand_mean"]
demand_std = checkpoint["demand_std"]
target_mean = checkpoint["target_mean"]
target_std = checkpoint["target_std"]

# =====================================================
# BUILD GRAPH
# =====================================================

edge_index = torch.tensor(
    edges_df[
        ["source_index", "target_index"]
    ].to_numpy().T.copy(),
    dtype=torch.long
)

edge_index = torch.cat(
    [edge_index, edge_index.flip(0)],
    dim=1
)

graph = Data(
    edge_index=edge_index,
    num_nodes=len(nodes_df)
)

# =====================================================
# SELECT LATEST OBSERVED HOUR
# =====================================================

latest_hour = training_df["pickup_hour"].max()

print("Latest observed input hour:", latest_hour)

hour_rows = training_df[
    training_df["pickup_hour"] == latest_hour
].copy()

# The hourly dataset contains only the 150 observed
# pickup zones. Initialize all 220 graph nodes to zero.
zone_to_index = {
    int(zone): int(index)
    for zone, index in zip(
        nodes_df["location_id"],
        nodes_df["node_index"]
    )
}

features = np.zeros(
    (len(nodes_df), 3),
    dtype=np.float32
)

for row in hour_rows.itertuples(index=False):
    zone = int(row.pickup_location)

    if zone not in zone_to_index:
        continue

    idx = zone_to_index[zone]

    features[idx] = [
        (float(row.trip_count) - demand_mean) / demand_std,
        np.sin(2 * np.pi * float(row.hour_of_day) / 24),
        np.cos(2 * np.pi * float(row.hour_of_day) / 24)
    ]

graph.x = torch.tensor(
    features,
    dtype=torch.float
).to(DEVICE)

graph.edge_index = graph.edge_index.to(DEVICE)

# =====================================================
# GENERATE PREDICTIONS
# =====================================================

print("Generating next-hour predictions...")

with torch.no_grad():
    normalized_predictions = model(graph).cpu().numpy()

predictions = np.maximum(
    normalized_predictions * target_std + target_mean,
    0
)

# Map graph node indices back to zone IDs.
results = pd.DataFrame({
    "pickup_location": nodes_df["location_id"].astype(int),
    "node_index": nodes_df["node_index"].astype(int),
    "input_hour": latest_hour,
    "prediction_hour": latest_hour + pd.Timedelta(hours=1),
    "predicted_demand": predictions
})

results["predicted_demand"] = results[
    "predicted_demand"
].round(3)

# =====================================================
# SAVE PREDICTIONS TO MINIO
# =====================================================

print("\nSample predictions:")
print(
    results.sort_values(
        "predicted_demand",
        ascending=False
    ).head(10).to_string(index=False)
)

results.to_parquet(
    OUTPUT_PATH,
    index=False,
    storage_options=STORAGE_OPTIONS
)

print("\nPredictions generated:", len(results))
print("Saved to:", OUTPUT_PATH)
print("Prediction hour:", latest_hour + pd.Timedelta(hours=1))
print("Prediction generation completed.")