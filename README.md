# Intelligent Self-Adaptive Data Orchestration for Real-Time Trip Analytics

## Project Overview

This project develops an intelligent data pipeline for trip analytics.

The baseline architecture is:

Kafka → MinIO → Spark → Snowflake

The proposed architecture adds:

Graph Construction → GNN → Anomaly/Demand Prediction → Decision Engine → Adaptive Airflow

The final system will provide a React-based dashboard.

## Technology Stack

- Apache Kafka
- MinIO
- Apache Spark
- Apache Airflow
- PyTorch Geometric
- Snowflake
- FastAPI
- React + TypeScript
- Docker

## Current Phase

Phase 1 — Project structure and infrastructure setup.

## Architecture

```text
Trip Data
    ↓
Kafka
    ↓
MinIO
    ↓
Spark
    ↓
Graph Builder
    ↓
GNN
    ↓
Decision Engine
    ↓
Airflow
    ↓
Snowflake
    ↓
FastAPI
    ↓
React