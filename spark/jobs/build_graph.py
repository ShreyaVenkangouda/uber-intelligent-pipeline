# pyrefly: ignore [missing-import]
from pyspark.sql import SparkSession
# pyrefly: ignore [missing-import]
from pyspark.sql.functions import (
    col,
    count,
    avg,
    sum
)

# --------------------------------------------------
# Spark session
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("UberGraphBuilder")
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
print("      UBER GRAPH CONSTRUCTION")
print("========================================\n")


# --------------------------------------------------
# Read processed data
# --------------------------------------------------

input_path = "s3a://uber-data/processed/uber_trips"

df = spark.read.parquet(input_path)

print("Input records:", df.count())

print("\nInput schema:")
df.printSchema()


# --------------------------------------------------
# 1. CREATE NODES
# --------------------------------------------------

pickup_nodes = (
    df.select(col("pickup_location").alias("location_id"))
)

dropoff_nodes = (
    df.select(col("dropoff_location").alias("location_id"))
)

nodes = (
    pickup_nodes
    .union(dropoff_nodes)
    .distinct()
)

# Calculate node-level features
outgoing = (
    df.groupBy("pickup_location")
    .agg(
        count("*").alias("outgoing_trips"),
        avg("fare_amount").alias("avg_fare"),
        avg("trip_distance").alias("avg_distance")
    )
    .withColumnRenamed("pickup_location", "location_id")
)

incoming = (
    df.groupBy("dropoff_location")
    .agg(
        count("*").alias("incoming_trips")
    )
    .withColumnRenamed("dropoff_location", "location_id")
)

nodes = (
    nodes
    .join(outgoing, on="location_id", how="left")
    .join(incoming, on="location_id", how="left")
    .fillna(0)
)

print("\nNumber of nodes:", nodes.count())

print("\nNodes:")
nodes.orderBy("location_id").show(20, truncate=False)


# --------------------------------------------------
# 2. CREATE DIRECTED EDGES
# --------------------------------------------------

edges = (
    df.groupBy(
        "pickup_location",
        "dropoff_location"
    )
    .agg(
        count("*").alias("trip_count"),
        avg("trip_distance").alias("avg_distance"),
        avg("fare_amount").alias("avg_fare"),
        avg("trip_duration_minutes").alias("avg_duration"),
        avg("surge_fee").alias("avg_surge")
    )
    .withColumnRenamed(
        "pickup_location",
        "source"
    )
    .withColumnRenamed(
        "dropoff_location",
        "target"
    )
)

print("\nNumber of edges:", edges.count())

print("\nEdges:")
edges.orderBy(
    col("trip_count").desc()
).show(20, truncate=False)


# --------------------------------------------------
# 3. SAVE GRAPH
# --------------------------------------------------

nodes_output = "s3a://uber-data/graph/nodes"
edges_output = "s3a://uber-data/graph/edges"

nodes.write.mode("overwrite").parquet(nodes_output)

edges.write.mode("overwrite").parquet(edges_output)


# --------------------------------------------------
# 4. FINAL SUMMARY
# --------------------------------------------------

print("\n========================================")
print("       GRAPH BUILD COMPLETE")
print("========================================")

print("Nodes saved to:")
print(nodes_output)

print("\nEdges saved to:")
print(edges_output)

print("\nNode count:", nodes.count())
print("Edge count:", edges.count())

print("\n========================================\n")

spark.stop()