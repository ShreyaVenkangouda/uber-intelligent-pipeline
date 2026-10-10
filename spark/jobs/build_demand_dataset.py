# pyrefly: ignore [missing-import]
from pyspark.sql import SparkSession
# pyrefly: ignore [missing-import]
from pyspark.sql import functions as F
# pyrefly: ignore [missing-import]
from pyspark.sql.window import Window

# =========================================================
# CONFIGURATION
# =========================================================

MINIO_ENDPOINT = "http://minio:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "admin12345"

INPUT_PATH = "s3a://uber-data/processed/uber_trips"
OUTPUT_PATH = "s3a://uber-data/demand/hourly"


# =========================================================
# CREATE SPARK SESSION WITH MINIO CONFIGURATION
# =========================================================

spark = (
    SparkSession.builder
    .appName("BuildHourlyDemandDataset")
    .config(
        "spark.hadoop.fs.s3a.endpoint",
        MINIO_ENDPOINT
    )
    .config(
        "spark.hadoop.fs.s3a.access.key",
        MINIO_ACCESS_KEY
    )
    .config(
        "spark.hadoop.fs.s3a.secret.key",
        MINIO_SECRET_KEY
    )
    .config(
        "spark.hadoop.fs.s3a.path.style.access",
        "true"
    )
    .config(
        "spark.hadoop.fs.s3a.impl",
        "org.apache.hadoop.fs.s3a.S3AFileSystem"
    )
    .config(
        "spark.hadoop.fs.s3a.connection.ssl.enabled",
        "false"
    )
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider"
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


try:
    # =====================================================
    # STEP 1: READ PROCESSED TRIPS
    # =====================================================

    print("\n==============================================")
    print("       BUILD HOURLY DEMAND DATASET")
    print("==============================================")

    print("\nReading processed Uber trips...")

    trips = spark.read.parquet(INPUT_PATH)

    print("Processed trip schema:")
    trips.printSchema()

    print(f"Total processed trips: {trips.count()}")


    # =====================================================
    # STEP 2: VALIDATE REQUIRED COLUMNS
    # =====================================================

    required_columns = {
        "pickup_location",
        "pickup_timestamp"
    }

    missing_columns = required_columns - set(trips.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    trips = trips.filter(
        F.col("pickup_location").isNotNull()
        & F.col("pickup_timestamp").isNotNull()
    )


    # =====================================================
    # STEP 3: AGGREGATE DEMAND BY ZONE AND HOUR
    # =====================================================

    print("\nAggregating trips by pickup zone and hour...")

    hourly_demand = (
        trips
        .withColumn(
            "pickup_hour",
            F.date_trunc(
                "hour",
                F.col("pickup_timestamp")
            )
        )
        .groupBy(
            "pickup_location",
            "pickup_hour"
        )
        .agg(
            F.count("*").alias("trip_count")
        )
    )

    print("\nSample hourly demand records:")

    hourly_demand.orderBy(
        "pickup_hour",
        "pickup_location"
    ).show(10, truncate=False)


    # =====================================================
    # STEP 4: ANALYZE THE OBSERVED TIME RANGE
    # =====================================================

    print("\nCalculating dataset summary...")

    hourly_demand.agg(
        F.min("pickup_hour").alias("first_hour"),
        F.max("pickup_hour").alias("last_hour"),
        F.countDistinct(
            "pickup_location"
        ).alias("pickup_zones"),
        F.countDistinct(
            "pickup_hour"
        ).alias("observed_hours")
    ).show(truncate=False)


    # =====================================================
    # STEP 5: IDENTIFY MISSING HOURS IN THE TIMELINE
    # =====================================================

    print("\nChecking observed hourly timestamps...")

    observed_hours = (
        hourly_demand
        .select("pickup_hour")
        .distinct()
        .orderBy("pickup_hour")
    )

    observed_hours.show(100, truncate=False)

    window_spec = Window.orderBy("pickup_hour")

    hour_gaps = (
        observed_hours
        .withColumn(
            "previous_hour",
            F.lag("pickup_hour").over(window_spec)
        )
        .withColumn(
            "gap_hours",
            (
                F.unix_timestamp("pickup_hour")
                - F.unix_timestamp("previous_hour")
            ) / 3600
        )
        .filter(
            F.col("gap_hours") > 1
        )
        .select(
            "previous_hour",
            "pickup_hour",
            "gap_hours"
        )
        .orderBy("previous_hour")
    )

    print("\nGaps between observed hours:")

    hour_gaps.show(100, truncate=False)


    # =====================================================
    # STEP 6: SAVE HOURLY DEMAND TO MINIO
    # =====================================================

    print("\nSaving hourly demand dataset to MinIO...")

    (
        hourly_demand
        .write
        .mode("overwrite")
        .parquet(OUTPUT_PATH)
    )

    print(f"\nOutput location: {OUTPUT_PATH}")
    print(f"Zone-hour records: {hourly_demand.count()}")

    print("\n==============================================")
    print(" HOURLY DEMAND DATASET COMPLETED SUCCESSFULLY")
    print("==============================================")


finally:
    spark.stop()