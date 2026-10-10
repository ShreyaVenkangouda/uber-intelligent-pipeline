# pyrefly: ignore [missing-import]
from pyspark.sql import SparkSession
# pyrefly: ignore [missing-import]
from pyspark.sql import functions as F

# =====================================================
# CONFIGURATION
# =====================================================

MINIO_ENDPOINT = "http://minio:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "admin12345"

INPUT_PATH = "s3a://uber-data/demand/hourly"
OUTPUT_PATH = "s3a://uber-data/demand/gnn_training"


# =====================================================
# SPARK SESSION
# =====================================================

spark = (
    SparkSession.builder
    .appName("PrepareDemandTraining")
    .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
    .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
    .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
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

try:
    # =================================================
    # STEP 1: LOAD HOURLY DEMAND
    # =================================================

    print("Loading hourly demand...")

    demand = spark.read.parquet(INPUT_PATH)

    # =================================================
    # STEP 2: BUILD COMPLETE ZONE-HOUR GRID
    # =================================================

    print("Building zone-hour grid...")

    hours = (
        demand
        .select("pickup_hour")
        .distinct()
    )

    zones = (
        demand
        .select("pickup_location")
        .distinct()
    )

    # Include every observed zone at every observed hour.
    complete_grid = zones.crossJoin(hours)

    complete_demand = (
        complete_grid
        .join(
            demand,
            on=["pickup_location", "pickup_hour"],
            how="left"
        )
        .withColumn(
            "trip_count",
            F.coalesce(F.col("trip_count"), F.lit(0))
        )
    )

    # =================================================
    # STEP 3: CREATE NEXT-HOUR TARGET
    # =================================================

    print("Creating next-hour prediction targets...")

    # The target for hour t is demand at hour t + 1.
    next_hour_demand = complete_demand.select(
        "pickup_location",
        F.col("pickup_hour").alias("target_hour"),
        F.col("trip_count").alias("target_demand")
    )

    training_data = (
        complete_demand
        .withColumn(
            "target_hour",
            F.expr("pickup_hour + INTERVAL 1 HOUR")
        )
        .join(
            next_hour_demand,
            on=["pickup_location", "target_hour"],
            how="inner"
        )
        .select(
            "pickup_location",
            "pickup_hour",
            "trip_count",
            "target_hour",
            "target_demand"
        )
        .withColumn(
            "hour_of_day",
            F.hour("pickup_hour")
        )
        .withColumn(
            "day_of_week",
            F.dayofweek("pickup_hour")
        )
        .withColumn(
            "target_demand",
            F.col("target_demand").cast("double")
        )
    )

    # =================================================
    # STEP 4: INSPECT THE DATASET
    # =================================================

    print("Sample training records:")

    training_data.orderBy(
        "pickup_hour",
        "pickup_location"
    ).show(10, truncate=False)

    print("Training dataset summary:")

    training_data.agg(
        F.count("*").alias("training_rows"),
        F.countDistinct("pickup_location").alias("zones"),
        F.countDistinct("pickup_hour").alias("input_hours"),
        F.min("pickup_hour").alias("first_input_hour"),
        F.max("pickup_hour").alias("last_input_hour")
    ).show(truncate=False)

    # =================================================
    # STEP 5: SAVE TO MINIO
    # =================================================

    print("Saving GNN training dataset...")

    (
        training_data
        .write
        .mode("overwrite")
        .parquet(OUTPUT_PATH)
    )

    print(f"Saved to: {OUTPUT_PATH}")
    print("Demand training preparation completed.")

finally:
    spark.stop()