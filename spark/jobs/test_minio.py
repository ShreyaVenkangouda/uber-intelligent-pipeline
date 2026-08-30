from pyspark.sql import SparkSession


# =========================================================
# Spark Session
# =========================================================

spark = (
    SparkSession.builder
    .appName("MinIOConnectionTest")

    # -----------------------------------------------------
    # MinIO / S3A Configuration
    # -----------------------------------------------------

    .config(
        "spark.hadoop.fs.s3a.endpoint",
        "http://minio:9000"
    )

    .config(
        "spark.hadoop.fs.s3a.access.key",
        "admin"
    )

    .config(
        "spark.hadoop.fs.s3a.secret.key",
        "admin12345"
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


# =========================================================
# Spark Logging
# =========================================================

spark.sparkContext.setLogLevel("WARN")


print("\n==============================================")
print("       SPARK → MINIO CONNECTION TEST")
print("==============================================")

print("\nReading data from MinIO...")


# =========================================================
# MinIO File Path
# =========================================================

path = (
    "s3a://uber-data/"
    "raw/20260830_144319/"
    "batch_0001.json"
)


print(f"\nInput file:")
print(path)


# =========================================================
# Read JSON
# =========================================================
#
# multiLine=True is required because our Kafka
# consumer stored each batch as a JSON array:
#
# [
#   {...},
#   {...}
# ]
#
# =========================================================

df = (
    spark.read
    .option("multiLine", "true")
    .json(path)
)


# =========================================================
# Success
# =========================================================

print("\n==============================================")
print("                 SUCCESS!")
print("==============================================")


# =========================================================
# Schema
# =========================================================

print("\nSchema:")
df.printSchema()


# =========================================================
# Record Count
# =========================================================

print("\nNumber of records:")

record_count = df.count()

print(record_count)


# =========================================================
# Sample Records
# =========================================================

print("\nSample records:")

df.show(
    5,
    truncate=False
)


# =========================================================
# Column Names
# =========================================================

print("\nColumns:")

for column in df.columns:
    print(f" - {column}")


# =========================================================
# Stop Spark
# =========================================================

spark.stop()

print("\n==============================================")
print("           TEST COMPLETED")
print("==============================================")