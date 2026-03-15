# MLOps Task 0: Batch Processing Pipeline

## Overview
This repository contains a production-ready Python pipeline that processes historical OHLCV data. It calculates a rolling mean and generates a binary trading signal. The pipeline is designed with MLOps best practices, including deterministic reproducibility, robust error handling, and containerization.

## Features & Requirements Fulfilled
- **Argparse:** No hardcoded paths; all inputs/outputs are passed via CLI.
- **Reproducibility:** A configuration file (`config.yaml`) sets a deterministic seed and parameters.
- **Data Validation:** Handles missing files, empty files, and hidden formatting traps (e.g., rogue quotes in CSV headers) safely.
- **Logging:** Comprehensive logging to both the console and `run.log`, starting with the exact job timestamp.
- **Metrics JSON:** Ensures `metrics.json` is generated in both success and error states.
- **Dockerized:** Fully portable and runs in an isolated `python:3.9-slim` container.

## Project Structure
- `run.py`: The core pipeline logic.
- `config.yaml`: Configuration parameters (seed, window size, version).
- `data.csv`: Sample input OHLCV dataset.
- `Dockerfile`: Container definition.
- `requirements.txt`: Python dependencies (`pandas`, `numpy`, `pyyaml`).

## How to Run locally (Python)
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
