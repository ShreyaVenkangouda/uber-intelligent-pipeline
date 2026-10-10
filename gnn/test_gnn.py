import pandas as pd
# pyrefly: ignore [missing-import]
import torch

# pyrefly: ignore [missing-import]
from torch_geometric.data import Data


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MINIO_ENDPOINT = "http://minio:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "admin12345"

NODES_PATH = "s3://uber-data/gnn/nodes"
EDGES_PATH = "s3://uber-data/gnn/edges"


# --------------------------------------------------
# Configure S3
# --------------------------------------------------

storage_options = {
    "key": MINIO_ACCESS_KEY,
    "secret": MINIO_SECRET_KEY,
    "client_kwargs": {
        "endpoint_url": MINIO_ENDPOINT
    }
}


print("========================================")
print("       LOAD GRAPH INTO PYTORCH")
print("========================================")


# --------------------------------------------------
# Read graph from MinIO
# --------------------------------------------------

nodes = pd.read_parquet(
    NODES_PATH,
    storage_options=storage_options
)

edges = pd.read_parquet(
    EDGES_PATH,
    storage_options=storage_options
)


print("\nNodes loaded:", len(nodes))
print("Edges loaded:", len(edges))


# --------------------------------------------------
# Create node features
# --------------------------------------------------

node_features = nodes[
    [
        "outgoing_trips",
        "incoming_trips",
        "avg_fare",
        "avg_distance"
    ]
].values


x = torch.tensor(
    node_features,
    dtype=torch.float
)


# --------------------------------------------------
# Create edge index
# --------------------------------------------------

edge_index = torch.tensor(
    edges[
        [
            "source_index",
            "target_index"
        ]
    ].values.T,
    dtype=torch.long
)


# --------------------------------------------------
# Create edge features
# --------------------------------------------------

edge_features = edges[
    [
        "trip_count",
        "avg_distance",
        "avg_fare",
        "avg_duration",
        "avg_surge"
    ]
].values


edge_attr = torch.tensor(
    edge_features,
    dtype=torch.float
)


# --------------------------------------------------
# Create PyTorch Geometric graph
# --------------------------------------------------

data = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr
)


# --------------------------------------------------
# Display graph information
# --------------------------------------------------

print("\n========================================")
print("       PYTORCH GEOMETRIC GRAPH")
print("========================================")

print("\nGraph:")
print(data)

print("\nNode feature shape:")
print(data.x.shape)

print("\nEdge index shape:")
print(data.edge_index.shape)

print("\nEdge feature shape:")
print(data.edge_attr.shape)

print("\nNumber of nodes:")
print(data.num_nodes)

print("\nNumber of edges:")
print(data.num_edges)

print("\nNode features:")
print(data.x[:5])

print("\nEdge index:")
print(data.edge_index[:, :5])

print("\nEdge features:")
print(data.edge_attr[:5])

print("\n========================================")
print("       GRAPH LOAD SUCCESSFUL")
print("========================================")