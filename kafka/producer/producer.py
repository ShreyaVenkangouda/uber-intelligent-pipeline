import csv
import json
import time
from pathlib import Path

from kafka import KafkaProducer


KAFKA_SERVER = "localhost:9092"
TOPIC_NAME = "uber-trips"

DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "raw" / "uber.csv"


def create_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_SERVER,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    )


def stream_csv():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    producer = create_producer()

    print(f"Reading dataset: {DATA_FILE}")
    print(f"Sending records to Kafka topic: {TOPIC_NAME}")
    print("Starting stream...\n")

    with open(DATA_FILE, "r", encoding="utf-8", newline="") as file:

        reader = csv.DictReader(file)

        for record_number, row in enumerate(reader, start=1):

            producer.send(
                TOPIC_NAME,
                value=row
            )

            print(f"Sent record {record_number}: {row}")

            # Simulate real-time streaming
            time.sleep(0.2)

    producer.flush()

    print("\nFinished streaming dataset.")


if __name__ == "__main__":
    stream_csv()