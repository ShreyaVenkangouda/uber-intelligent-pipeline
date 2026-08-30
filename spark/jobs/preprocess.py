from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    to_timestamp,
    hour,
    dayofweek,
    unix_timestamp,
    when,
    round,
    count
)


# =========================================================
# CONFIGURATION
# =========================================================

MINIO_ENDPOINT = "http://minio:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "admin12345"

INPUT_PATH = "s3a://uber-data/raw/*/*.json"
OUTPUT_PATH = "s3a://uber-data/processed/uber_trips"


# =========================================================
# CREATE SPARK SESSION
# =========================================================

spark = (
    SparkSession.builder
    .appName("UberTripPreprocessing")

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


print("\n==============================================")
print("       UBER DATA PREPROCESSING")
print("==============================================")


# =========================================================
# READ RAW DATA
# =========================================================

print("\nReading all raw batches from MinIO...")

df = (
    spark.read
    .option("multiLine", "true")
    .json(INPUT_PATH)
)


print("\nRaw schema:")
df.printSchema()


raw_count = df.count()

print(f"\nRaw record count: {raw_count}")


# =========================================================
# REMOVE DUPLICATES
# =========================================================

print("\nRemoving duplicate records...")

df = df.dropDuplicates(["Trip ID"])

after_duplicate_count = df.count()

print(
    f"Records after duplicate removal: "
    f"{after_duplicate_count}"
)


# =========================================================
# CONVERT DATA TYPES
# =========================================================

print("\nConverting data types...")


df = (
    df
    .withColumn(
        "pickup_location",
        col("PULocationID").cast("integer")
    )

    .withColumn(
        "dropoff_location",
        col("DOLocationID").cast("integer")
    )

    .withColumn(
        "fare_amount",
        col("fare_amount").cast("double")
    )

    .withColumn(
        "trip_distance",
        col("trip_distance").cast("double")
    )

    .withColumn(
        "passenger_count",
        col("passenger_count").cast("integer")
    )

    .withColumn(
        "surge_fee",
        col("Surge Fee").cast("double")
    )

    .withColumn(
        "pickup_timestamp",
        to_timestamp(
            col("Pickup Time"),
            "M/d/yyyy H:mm"
        )
    )

    .withColumn(
        "dropoff_timestamp",
        to_timestamp(
            col("Drop Off Time"),
            "M/d/yyyy H:mm"
        )
    )
)


# =========================================================
# TRIP DURATION
# =========================================================

print("\nCalculating trip duration...")


df = df.withColumn(
    "trip_duration_minutes",
    (
        unix_timestamp("dropoff_timestamp")
        - unix_timestamp("pickup_timestamp")
    ) / 60
)


# =========================================================
# TEMPORAL FEATURES
# =========================================================

print("\nCreating temporal features...")


df = (
    df

    .withColumn(
        "pickup_hour",
        hour("pickup_timestamp")
    )

    .withColumn(
        "pickup_day_of_week",
        dayofweek("pickup_timestamp")
    )
)


# =========================================================
# FARE PER DISTANCE
# =========================================================

print("\nCreating fare efficiency feature...")


df = df.withColumn(
    "fare_per_distance",
    when(
        col("trip_distance") > 0,
        round(
            col("fare_amount") / col("trip_distance"),
            2
        )
    ).otherwise(None)
)


# =========================================================
# CLEAN INVALID VALUES
# =========================================================

print("\nRemoving invalid trips...")


df = df.filter(
    col("pickup_location").isNotNull()
)

df = df.filter(
    col("dropoff_location").isNotNull()
)

df = df.filter(
    col("pickup_timestamp").isNotNull()
)

df = df.filter(
    col("dropoff_timestamp").isNotNull()
)

df = df.filter(
    (col("trip_distance").isNull()) |
    (col("trip_distance") >= 0)
)

df = df.filter(
    (col("fare_amount").isNull()) |
    (col("fare_amount") >= 0)
)

df = df.filter(
    (col("passenger_count").isNull()) |
    (col("passenger_count") > 0)
)


# =========================================================
# SELECT FINAL COLUMNS
# =========================================================

processed_df = df.select(
    "Trip ID",
    "pickup_location",
    "dropoff_location",
    "pickup_timestamp",
    "dropoff_timestamp",
    "trip_duration_minutes",
    "fare_amount",
    "trip_distance",
    "passenger_count",
    "surge_fee",
    "Vehicle",
    "Payment_type",
    "pickup_hour",
    "pickup_day_of_week",
    "fare_per_distance"
)


# =========================================================
# SHOW RESULT
# =========================================================

print("\n==============================================")
print("             PROCESSED DATA")
print("==============================================")


processed_df.printSchema()


final_count = processed_df.count()

print(
    f"\nFinal record count: {final_count}"
)


print("\nSample processed records:")

processed_df.show(
    10,
    truncate=False
)


# =========================================================
# WRITE TO MINIO
# =========================================================

print("\nWriting processed data to MinIO...")


(
    processed_df
    .write
    .mode("overwrite")
    .parquet(OUTPUT_PATH)
)


print("\n==============================================")
print("       PREPROCESSING COMPLETED")
print("==============================================")

print(
    f"\nOutput location:\n{OUTPUT_PATH}"
)


spark.stop()