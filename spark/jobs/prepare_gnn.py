# pyrefly: ignore [missing-import]
from pyspark.sql import SparkSession
# pyrefly: ignore [missing-import]
from pyspark.sql.functions import col, row_number
# pyrefly: ignore [missing-import]
from pyspark.sql.window import Window


# --------------------------------------------------
# Spark session
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("PrepareGNNData")
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
    .config("spark.hadoop.fs.s3a.access.key", "admin")
    .config("spark.hadoop.fs.s3a.secret.key", "admin12345")
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config(
        "spark.hadoop.fs.s3a.impl",
        "org.apache.hadoop.fs.s3a.S3AFileSystem"
    )
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider"
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("\n========================================")
print("        PREPARE GNN DATA")
print("========================================\n")


# --------------------------------------------------
# Read graph
# --------------------------------------------------

nodes_path = "s3a://uber-data/graph/nodes"
edges_path = "s3a://uber-data/graph/edges"

nodes = spark.read.parquet(nodes_path)
edges = spark.read.parquet(edges_path)

print("Original nodes:", nodes.count())
print("Original edges:", edges.count())


# --------------------------------------------------
# Create continuous node indices
# --------------------------------------------------

window = Window.orderBy("location_id")

nodes = (
    nodes
    .withColumn(
        "node_index",
        row_number().over(window) - 1
    )
)

print("\nNode mapping:")
nodes.select(
    "location_id",
    "node_index"
).orderBy("node_index").show(20)


# --------------------------------------------------
# Add source node index to edges
# --------------------------------------------------

source_mapping = (
    nodes
    .select(
        col("location_id").alias("source_location"),
        col("node_index").alias("source_index")
    )
)

target_mapping = (
    nodes
    .select(
        col("location_id").alias("target_location"),
        col("node_index").alias("target_index")
    )
)


edges = (
    edges
    .join(
        source_mapping,
        edges.source == source_mapping.source_location,
        "left"
    )
    .join(
        target_mapping,
        edges.target == target_mapping.target_location,
        "left"
    )
)


# --------------------------------------------------
# Select GNN edge columns
# --------------------------------------------------

edges = edges.select(
    "source",
    "target",
    "source_index",
    "target_index",
    "trip_count",
    "avg_distance",
    "avg_fare",
    "avg_duration",
    "avg_surge"
)


# --------------------------------------------------
# Verify mappings
# --------------------------------------------------

missing_edges = edges.filter(
    col("source_index").isNull() |
    col("target_index").isNull()
).count()

print("\nEdges with missing node mapping:", missing_edges)


# --------------------------------------------------
# Save GNN-ready data
# --------------------------------------------------

gnn_nodes_path = "s3a://uber-data/gnn/nodes"
gnn_edges_path = "s3a://uber-data/gnn/edges"

nodes.write.mode("overwrite").parquet(gnn_nodes_path)
edges.write.mode("overwrite").parquet(gnn_edges_path)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n========================================")
print("       GNN DATA PREPARATION COMPLETE")
print("========================================")

print("\nGNN nodes:")
print(gnn_nodes_path)

print("\nGNN edges:")
print(gnn_edges_path)

print("\nNumber of nodes:", nodes.count())
print("Number of edges:", edges.count())

print("\nSample GNN edges:")
edges.show(10, truncate=False)

print("\n========================================\n")

spark.stop()