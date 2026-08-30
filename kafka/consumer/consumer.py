import json
import io
from datetime import datetime

from kafka import KafkaConsumer
# pyrefly: ignore [missing-import]
from minio import Minio


# =========================
# Kafka Configuration
# =========================

KAFKA_SERVER = "localhost:9092"
TOPIC_NAME = "uber-trips"
GROUP_ID = "uber-minio-consumer"


# =========================
# MinIO Configuration
# =========================

MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "admin12345"

BUCKET_NAME = "uber-data"


# =========================
# Batch Configuration
# =========================

BATCH_SIZE = 100


# =========================
# Create Kafka Consumer
# =========================

consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers=KAFKA_SERVER,
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id=GROUP_ID,
    value_deserializer=lambda value: json.loads(
        value.decode("utf-8")
    ),
)


# =========================
# Create MinIO Client
# =========================

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)


# =========================
# Create Bucket If Needed
# =========================

if not minio_client.bucket_exists(BUCKET_NAME):
    minio_client.make_bucket(BUCKET_NAME)
    print(f"Created bucket: {BUCKET_NAME}")
else:
    print(f"Bucket already exists: {BUCKET_NAME}")


# =========================
# Save Batch To MinIO
# =========================

def save_batch(batch, batch_number):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    object_name = (
        f"raw/{timestamp}/batch_{batch_number:04d}.json"
    )

    data = json.dumps(
        batch,
        indent=2
    ).encode("utf-8")

    data_stream = io.BytesIO(data)

    minio_client.put_object(
        BUCKET_NAME,
        object_name,
        data_stream,
        length=len(data),
        content_type="application/json",
    )

    print(
        f"Saved batch {batch_number} "
        f"({len(batch)} records) → "
        f"{BUCKET_NAME}/{object_name}"
    )


# =========================
# Consume Kafka Messages
# =========================

print("\nKafka → MinIO consumer started")
print(f"Topic: {TOPIC_NAME}")
print(f"Batch size: {BATCH_SIZE}\n")


batch = []
batch_number = 1


try:

    for message in consumer:

        batch.append(message.value)

        print(
            f"Received record "
            f"{len(batch)}/{BATCH_SIZE}"
        )

        if len(batch) >= BATCH_SIZE:

            save_batch(
                batch,
                batch_number
            )

            batch = []
            batch_number += 1


except KeyboardInterrupt:

    print("\nStopping consumer...")

    # Save remaining records
    if batch:
        save_batch(
            batch,
            batch_number
        )

    print("Consumer stopped.")