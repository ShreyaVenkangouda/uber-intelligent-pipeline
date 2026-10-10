import os
import copy
import random
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

MODEL_DIR = "/app/models"
MODEL_PATH = os.path.join(MODEL_DIR, "demand_gcn.pt")

SEED = 42
EPOCHS = 150
LEARNING_RATE = 0.01
HIDDEN_CHANNELS = 32

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")

STORAGE_OPTIONS = {
    "key": MINIO_ACCESS_KEY,
    "secret": MINIO_SECRET_KEY,
    "client_kwargs": {
        "endpoint_url": MINIO_ENDPOINT
    }
}


# =====================================================
# LOAD DATA FROM MINIO
# =====================================================

print("\nLoading training dataset...")

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

training_df["target_hour"] = pd.to_datetime(
    training_df["target_hour"]
)

nodes_df = nodes_df.sort_values(
    "node_index"
).reset_index(drop=True)

print("Training rows:", len(training_df))
print("Graph nodes:", len(nodes_df))
print("Graph edges:", len(edges_df))


# =====================================================
# BUILD GRAPH
# =====================================================

print("\nBuilding graph...")

num_nodes = len(nodes_df)

# The graph uses pickup -> dropoff relationships.
edge_index = torch.tensor(
    edges_df[
        ["source_index", "target_index"]
    ].to_numpy().T.copy(),
    dtype=torch.long
)

# Make the graph usable in both directions.
edge_index = torch.cat(
    [edge_index, edge_index.flip(0)],
    dim=1
)

# Add self-loops through GCNConv's default behavior.
graph = Data(
    edge_index=edge_index,
    num_nodes=num_nodes
).to(DEVICE)

print("Graph:", graph)
print("Edge index shape:", graph.edge_index.shape)


# =====================================================
# PREPARE TEMPORAL TRAIN / VALIDATION / TEST SPLITS
# =====================================================

print("\nPreparing chronological data splits...")

hours = sorted(
    training_df["pickup_hour"].unique()
)

if len(hours) < 10:
    raise ValueError(
        "Not enough hourly observations for a meaningful split."
    )

train_end = int(len(hours) * 0.70)
val_end = int(len(hours) * 0.85)

train_hours = set(hours[:train_end])
val_hours = set(hours[train_end:val_end])
test_hours = set(hours[val_end:])

print("Training hours:", len(train_hours))
print("Validation hours:", len(val_hours))
print("Test hours:", len(test_hours))


# =====================================================
# NORMALIZATION
# =====================================================

# Fit scaling only on training hours.
train_rows = training_df[
    training_df["pickup_hour"].isin(train_hours)
]

demand_mean = float(train_rows["trip_count"].mean())
demand_std = float(train_rows["trip_count"].std())

if not np.isfinite(demand_std) or demand_std < 1e-6:
    demand_std = 1.0

target_mean = float(train_rows["target_demand"].mean())
target_std = float(train_rows["target_demand"].std())

if not np.isfinite(target_std) or target_std < 1e-6:
    target_std = 1.0

print("Demand normalization mean:", demand_mean)
print("Demand normalization std:", demand_std)


# =====================================================
# CONVERT EACH HOUR TO A GRAPH LEARNING EXAMPLE
# =====================================================

def make_example(hour_value):
    """Create node features and next-hour labels for one timestamp."""

    rows = (
        training_df[
            training_df["pickup_hour"] == hour_value
        ]
        .sort_values("pickup_location")
        .copy()
    )

    # Map rows into graph node-index order.
    zone_to_index = {
        int(zone): int(index)
        for zone, index in zip(
            nodes_df["location_id"],
            nodes_df["node_index"]
        )
    }

    features = np.zeros(
        (num_nodes, 3),
        dtype=np.float32
    )

    targets = np.zeros(
        num_nodes,
        dtype=np.float32
    )

    mask = np.zeros(
        num_nodes,
        dtype=bool
    )

    for row in rows.itertuples(index=False):
        zone = int(row.pickup_location)

        if zone not in zone_to_index:
            continue

        idx = zone_to_index[zone]

        normalized_demand = (
            float(row.trip_count) - demand_mean
        ) / demand_std

        hour_of_day = float(row.hour_of_day)
        day_of_week = float(row.day_of_week)

        features[idx] = [
            normalized_demand,
            np.sin(2 * np.pi * hour_of_day / 24),
            np.cos(2 * np.pi * hour_of_day / 24)
        ]

        targets[idx] = (
            float(row.target_demand) - target_mean
        ) / target_std

        mask[idx] = True

    data = Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=graph.edge_index,
        y=torch.tensor(targets, dtype=torch.float),
        mask=torch.tensor(mask, dtype=torch.bool)
    )

    return data.to(DEVICE)


train_examples = [
    make_example(hour)
    for hour in sorted(train_hours)
]

val_examples = [
    make_example(hour)
    for hour in sorted(val_hours)
]

test_examples = [
    make_example(hour)
    for hour in sorted(test_hours)
]

print("Training examples:", len(train_examples))
print("Validation examples:", len(val_examples))
print("Test examples:", len(test_examples))


# =====================================================
# GCN MODEL
# =====================================================

class DemandGCN(nn.Module):
    def __init__(self, hidden_channels=32):
        super().__init__()

        self.conv1 = GCNConv(
            3,
            hidden_channels
        )

        self.conv2 = GCNConv(
            hidden_channels,
            hidden_channels
        )

        self.output_layer = nn.Linear(
            hidden_channels,
            1
        )

    def forward(self, data):
        x = self.conv1(
            data.x,
            data.edge_index
        )

        x = F.relu(x)
        x = F.dropout(
            x,
            p=0.2,
            training=self.training
        )

        x = self.conv2(
            x,
            data.edge_index
        )

        x = F.relu(x)

        return self.output_layer(x).squeeze(-1)


model = DemandGCN(
    hidden_channels=HIDDEN_CHANNELS
).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

best_val_loss = float("inf")
best_state = None


# =====================================================
# EVALUATION HELPERS
# =====================================================

def evaluate(examples):
    model.eval()

    actual_values = []
    predicted_values = []

    with torch.no_grad():
        for data in examples:
            prediction = model(data)

            mask = data.mask

            predicted = (
                prediction[mask].cpu().numpy()
                * target_std
                + target_mean
            )

            actual = (
                data.y[mask].cpu().numpy()
                * target_std
                + target_mean
            )

            predicted_values.extend(
                np.maximum(predicted, 0).tolist()
            )

            actual_values.extend(
                actual.tolist()
            )

    if not actual_values:
        return float("nan"), float("nan")

    actual_values = np.asarray(actual_values)
    predicted_values = np.asarray(predicted_values)

    mae = np.mean(
        np.abs(actual_values - predicted_values)
    )

    rmse = np.sqrt(
        np.mean(
            (actual_values - predicted_values) ** 2
        )
    )

    return float(mae), float(rmse)


# =====================================================
# TRAIN
# =====================================================

print("\nStarting GCN training...")

for epoch in range(1, EPOCHS + 1):
    model.train()

    total_loss = 0.0
    batches = 0

    for data in train_examples:
        optimizer.zero_grad()

        prediction = model(data)

        loss = F.mse_loss(
            prediction[data.mask],
            data.y[data.mask]
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()
        batches += 1

    train_loss = total_loss / max(batches, 1)

    model.eval()

    val_losses = []

    with torch.no_grad():
        for data in val_examples:
            prediction = model(data)

            val_loss = F.mse_loss(
                prediction[data.mask],
                data.y[data.mask]
            )

            val_losses.append(val_loss.item())

    val_loss = (
        float(np.mean(val_losses))
        if val_losses
        else train_loss
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(
            model.state_dict()
        )

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Validation loss: {val_loss:.4f}"
        )


# =====================================================
# RESTORE BEST MODEL AND TEST
# =====================================================

if best_state is not None:
    model.load_state_dict(best_state)

test_mae, test_rmse = evaluate(test_examples)

print("\nFinal test results")
print("------------------")
print(f"Test MAE:  {test_mae:.4f} trips")
print(f"Test RMSE: {test_rmse:.4f} trips")


# =====================================================
# SAVE MODEL AND METADATA
# =====================================================

os.makedirs(MODEL_DIR, exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "num_nodes": num_nodes,
        "input_features": 3,
        "hidden_channels": HIDDEN_CHANNELS,
        "demand_mean": demand_mean,
        "demand_std": demand_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "zone_ids": nodes_df["location_id"].tolist()
    },
    MODEL_PATH
)

print("\nModel saved to:", MODEL_PATH)
print("GNN training completed successfully.")